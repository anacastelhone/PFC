"""
MODELOS -- tudo que envolve os detectores do projeto, num arquivo so
(substitui treinar.py e termico.py).

  PARTE 1  YOLO      deteccao dos componentes (InsPLAD-det): treino, validacao, carregamento
  PARTE 2  PATCHCORE anomalia fisica por componente (InsPLAD-fault): treino, carregamento
  PARTE 3  ITADA     hotspot no termico (Carpi & Lages, 2010)
  PARTE 4  HOMOGRAFIA RGB -> termico (leva a caixa do YOLO para o quadro termico)

Componentes (classes do YOLO, devem bater com o data.yaml):
    glass-insulator, polymer-insulator, vari-grip, spacer, stockbridge-damper

INSTALACAO (so o que for usar):
    pip install ultralytics          # yolo / validar
    pip install anomalib             # patchcore

USO:
    python modelos.py yolo dataset_det/data.yaml [--model yolov8n.pt --epochs 100 --device 0|cpu]
    python modelos.py validar runs_yolo/treino/weights/best.pt dataset_det/data.yaml
    python modelos.py patchcore insplad_fault/glass-insulator --name glass-insulator
    python modelos.py itada <pasta_com_imagens_termicas> [--saida overlays/]
    python modelos.py calibrar pontos_calibracao.json homografia_c12.npy

Bibliotecas pesadas (ultralytics, anomalib, torch) so sao importadas quando usadas:
a parte termica roda so com numpy + opencv.

TERMICO: a C12 deve estar em WHITE-HOT (tons de cinza, quente = claro). O ITADA usa o
brilho como temperatura; com paleta colorida (arco-iris, ferro) o resultado nao vale.
"""
import os
import json
import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import cv2

EXT_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp")


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread falha com acento no caminho (Windows): le os bytes e decodifica."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)
    return ok


# =============================================================================
# PARTE 1 -- YOLO (deteccao dos componentes)
# =============================================================================
@dataclass
class Detection:
    cls_name: str
    box: tuple          # (x1, y1, x2, y2) em pixels do quadro RGB
    conf: float


def treinar_yolo(data_yaml, model="yolov8n.pt", epochs=100, imgsz=640, batch=16, device="0",
                 project="runs_yolo", name="treino"):
    from ultralytics import YOLO
    YOLO(model).train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        patience=20, project=project, name=name,
        # augmentation: dataset pequeno + gap de dominio (drone -> robo).
        # Sem rotacao: em caixas, girar a imagem infla a caixa e piora o rotulo.
        degrees=0.0, translate=0.1, scale=0.3, fliplr=0.5, mosaic=1.0,
    )
    print(f"\nMelhores pesos em: {project}/{name}/weights/best.pt")


def validar_yolo(weights, data_yaml, imgsz=640, device="0", saida="validacao_yolo"):
    """Valida e imprime/salva P, R, mAP50 e mAP50-95 POR CLASSE."""
    import csv
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data_yaml, imgsz=imgsz, device=device, plots=True,
                          project=saida, name="val", exist_ok=True)
    box, nt = m.box, getattr(m, "nt_per_class", None)
    linhas = []
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        linhas.append({"classe": m.names[int(ci)], "instancias": int(nt[ci]) if nt is not None else "",
                       "precisao": round(float(p), 4), "recall": round(float(r), 4),
                       "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)})
    linhas.append({"classe": "TODAS", "instancias": sum(l["instancias"] or 0 for l in linhas),
                   "precisao": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
                   "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4)})
    os.makedirs(saida, exist_ok=True)
    caminho = os.path.join(saida, "metricas_por_classe.csv")
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\n{'classe':22s} {'inst':>6s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for l in linhas:
        print(f"{l['classe']:22s} {str(l['instancias']):>6s} {l['precisao']:7.3f} {l['recall']:7.3f} "
              f"{l['mAP50']:7.3f} {l['mAP50_95']:9.3f}")
    print(f"\nSalvo em {caminho}")
    return m


def carregar_detector(weights: str, conf: float = 0.25, classes: Optional[set] = None) -> Callable:
    """-> detect(frame_bgr) -> [Detection]. `classes` filtra pelo nome (None = todas)."""
    from ultralytics import YOLO
    yolo = YOLO(weights)

    def detect(frame_bgr: np.ndarray) -> list:
        r = yolo.predict(frame_bgr, conf=conf, verbose=False)[0]
        out = []
        for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            nome = r.names[int(c)]
            if not classes or nome in classes:
                out.append(Detection(nome, tuple(xyxy), float(s)))
        return out

    return detect


# =============================================================================
# PARTE 2 -- PATCHCORE (anomalia fisica, um modelo por componente)
# =============================================================================
# Usa a pasta de UMA classe do InsPLAD-fault direto, no formato MVTec:
#     <classe>/train/good/          so normais (treino)
#     <classe>/test/good/           normais de teste
#     <classe>/test/<tipo_defeito>/ uma pasta por tipo de defeito (todas viram "anomalia")
CROP_SIZE = 256


def _tem_imagem(pasta: str) -> bool:
    return os.path.isdir(pasta) and any(f.lower().endswith(EXT_IMAGEM) for f in os.listdir(pasta))


def pastas_defeito(pasta_classe: str) -> list:
    """Subpastas de test/ que nao sao 'good' e tem imagem (os tipos de defeito)."""
    test = os.path.join(pasta_classe, "test")
    if not os.path.isdir(test):
        return []
    return [d for d in sorted(os.listdir(test)) if d != "good" and _tem_imagem(os.path.join(test, d))]


def treinar_patchcore(pasta_classe: str, nome: str, batch_size: int = 8, coreset: float = 0.1,
                      saida: Optional[str] = None):
    """coreset: fracao do banco de memoria mantida. Se a RAM estourar (Colab), diminua
    (ex.: 0.05) ou deixe menos imagens em train/good."""
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    saida = saida or f"patchcore_{nome}"
    kwargs = dict(name=nome, root=pasta_classe, normal_dir="train/good",
                  train_batch_size=batch_size, eval_batch_size=batch_size, num_workers=4)
    defeitos = pastas_defeito(pasta_classe)
    if defeitos:
        kwargs["abnormal_dir"] = [os.path.join("test", d) for d in defeitos]
        if _tem_imagem(os.path.join(pasta_classe, "test", "good")):
            kwargs["normal_test_dir"] = "test/good"
        print(f"[{nome}] tipos de defeito no teste: {', '.join(defeitos)}")
    else:
        # o anomalib da erro ("Found 0 ABNORMAL images") com abnormal_dir vazio
        print(f"[AVISO] {pasta_classe}/test sem pastas de defeito: treinando sem avaliacao "
              f"contra anomalia conhecida.")

    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=coreset)
    engine = Engine(default_root_dir=saida, max_epochs=1)  # sem gradiente: 1 passada monta o banco
    engine.fit(model=model, datamodule=Folder(**kwargs))
    if defeitos:
        engine.test(model=model, datamodule=Folder(**kwargs))
    ckpt = getattr(engine, "best_model_path", None)
    print(f"\n[{nome}] treino concluido. Checkpoint: {ckpt or f'procure model.ckpt dentro de {saida}/'}")
    return engine, model


def recorte_quadrado(img: np.ndarray, box: tuple, margem: float = 0.0) -> np.ndarray:
    """Recorte QUADRADO centrado na caixa, de lado = maior lado da caixa * (1 + margem), como
    os recortes do InsPLAD-fault (componente comprido encosta nas bordas, sobra fundo dos lados).
    Assim o resize para 256x256 nao deforma o componente. Perto da borda da imagem o quadrado
    e deslocado para dentro (em vez de cortado), mantendo a proporcao."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    lado = int(round(max(x2 - x1, y2 - y1) * (1 + margem)))
    lado = max(1, min(lado, w, h))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(round(min(max(cx - lado / 2, 0), w - lado)))
    y0 = int(round(min(max(cy - lado / 2, 0), h - lado)))
    return img[y0:y0 + lado, x0:x0 + lado]


def carregar_modelo_patchcore(ckpt: str):
    """-> (model, device, transform). Base de carregar_patchcore e do avaliar.py."""
    import torch
    import torchvision.transforms.v2 as T
    from anomalib.models import Patchcore

    # PyTorch 2.6+ usa weights_only=True por padrao, o que quebra checkpoints do anomalib.
    # False e seguro aqui: o checkpoint e seu.
    model = Patchcore.load_from_checkpoint(ckpt, weights_only=False)
    model.eval()
    try:   # a entrada vai para o MESMO dispositivo do modelo (GPU no Colab)
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(model, "device", "cpu")
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                           T.Resize((CROP_SIZE, CROP_SIZE))])
    return model, device, transform


def inferir_patchcore(pc, img_bgr: np.ndarray):
    """pc = carregar_modelo_patchcore(...). -> (score, mapa_de_anomalia 2D ou None)."""
    import torch
    model, device, transform = pc
    x = transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    amap = getattr(out, "anomaly_map", None)
    return float(out.pred_score), (amap.squeeze().cpu().numpy() if amap is not None else None)


def carregar_patchcore(ckpt: str, margem: float = 0.0) -> Callable:
    """-> score(frame_bgr, box) -> float. Recorte quadrado (ver recorte_quadrado)."""
    pc = carregar_modelo_patchcore(ckpt)

    def score(frame_bgr: np.ndarray, box: tuple) -> float:
        crop = recorte_quadrado(frame_bgr, box, margem)
        return inferir_patchcore(pc, crop)[0] if crop.size else 0.0

    return score


def carregar_patchcores(mapa: dict, margem: float = 0.0) -> dict:
    """{classe_yolo: caminho.ckpt} -> {classe_yolo: score}. Classes sem checkpoint (ex.:
    spacer, stockbridge-damper, que nao tem pasta de falha no InsPLAD) ficam so com a deteccao."""
    return {cls: carregar_patchcore(ck, margem) for cls, ck in mapa.items()}


# =============================================================================
# PARTE 3 -- ITADA (Infrared Thermography Anomaly Detection Algorithm)
# =============================================================================
# Carpi & Lages (2010), "Robotized Inspection of Power Lines with Infrared Vision"
# (baseado em Chou & Yao, 2009).
#   Eq.(1)-(2) -> hottest_pixels_mask()   pixels de valor maximo (semente)
#   Eq.(3)     -> grow_hotspots()         dilatacao restrita ao gradiente C entre vizinhos
#   Eq.(4)     -> Hotspot.t_hot           media dos pixels do hotspot
#   Eq.(5)-(8) -> reference_stats()       T_ref = media do fundo; delta_T = T_hot - T_ref
#
# ADAPTACAO PARA A SKYDROID C12: o artigo usa imagem RADIOMETRICA (temperatura por pixel).
# A C12 entrega so imagem visual 8-bit, entao "temperatura" aqui = intensidade (maior =
# mais quente) e delta_T sai em NIVEIS DE CINZA. Por isso as Tabelas II/III do artigo
# (classificacao por graus C) nao sao usadas.
#
# Adaptacoes de engenharia (fora do artigo):
#   - roi_mask: Otsu e T_ref calculados so dentro da caixa do componente;
#   - limiar de contraste ADAPTATIVO: delta_T >= max(MIN_CONTRAST_FLOOR, MIN_CONTRAST_SIGMAS * desvio
#     do fundo). Um limiar fixo gerava falso positivo em cascata em fundo com textura;
#   - trava de area (max_area_fraction) contra regiao que "vaza" pela imagem sem hotspot real.
MIN_HOTSPOT_PIXELS = 5       # areas menores sao ruido
GRADIENT_CONSTRAINT_C = 16   # limite de gradiente entre vizinhos (Eq. 3)
MIN_CONTRAST_SIGMAS = 4.0
MIN_CONTRAST_FLOOR = 5.0


@dataclass
class Hotspot:
    mask: np.ndarray        # mascara booleana (mesmo tamanho da imagem)
    n_pixels: int
    t_hot: float            # Eq.(4)
    centroid: tuple         # (x, y) em pixels


@dataclass
class ITADAResult:
    hotspots: list          # ordenados do maior para o menor
    t_ref: float            # Eq.(6)
    delta_t: float          # Eq.(5) do maior hotspot; 0.0 se nao houver


def otsu_threshold(alpha: np.ndarray) -> float:
    """Limiar de Otsu na escala original de alpha. Aceita 2D ou 1D (ex.: alpha[roi])."""
    flat = alpha.reshape(-1, 1) if alpha.ndim == 1 else alpha
    img8 = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t_norm, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a_min, a_max = float(alpha.min()), float(alpha.max())
    return a_min + (t_norm / 255.0) * (a_max - a_min)


def segment(alpha: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """gamma(x,y) = alpha(x,y) se alpha >= T (Otsu), senao 0. Com roi_mask, o Otsu e
    calculado so dentro da ROI e tudo fora dela e zerado."""
    if roi_mask is None:
        return np.where(alpha >= otsu_threshold(alpha), alpha, 0.0)
    t = otsu_threshold(alpha[roi_mask])
    return np.where(roi_mask & (alpha >= t), alpha, 0.0)


def hottest_pixels_mask(gamma: np.ndarray) -> np.ndarray:
    t_hot = gamma.max()
    return np.zeros_like(gamma, bool) if t_hot == 0 else gamma == t_hot


def grow_hotspots(gamma: np.ndarray, seed_mask: np.ndarray, gradient_c: Optional[float] = None,
                  max_iter: int = 200, max_area_fraction: float = 0.05,
                  total_px: Optional[int] = None) -> np.ndarray:
    """Eq.(3): dilatacao iterativa aceitando o vizinho cuja diferenca para o pixel mais
    proximo JA na regiao e <= C (restricao LOCAL; contra o maximo global a regiao vaza).
    Se a regiao passar de max_area_fraction da imagem, devolve vazio (nao e hotspot fisico:
    fundo uniforme, plato de quantizacao 8-bit etc.). Trabalha numa janela em volta da
    regiao (mesmo resultado, bem mais rapido com muitas sementes)."""
    gradient_c = GRADIENT_CONSTRAINT_C if gradient_c is None else gradient_c   # lido na hora (ajustavel)
    mask = seed_mask.copy()
    max_pixels = int(max_area_fraction * (total_px or gamma.size))   # fracao da imagem INTEIRA
    if not mask.any() or mask.sum() > max_pixels:
        return np.zeros_like(mask, bool)

    kernel = np.ones((3, 3), np.uint8)
    h, w = gamma.shape
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    n_pix = int(mask.sum())

    for _ in range(max_iter):
        wy0, wy1, wx0, wx1 = max(0, y0 - 1), min(h, y1 + 1), max(0, x0 - 1), min(w, x1 + 1)
        sub_mask, sub_g = mask[wy0:wy1, wx0:wx1], gamma[wy0:wy1, wx0:wx1]
        borda = cv2.dilate(sub_mask.astype(np.uint8), kernel).astype(bool) & ~sub_mask
        if not borda.any():
            break
        vizinho = cv2.dilate(np.where(sub_mask, sub_g, -np.inf).astype(np.float32), kernel)
        aceita = borda & (sub_g > 0) & (np.abs(sub_g - vizinho) <= gradient_c)
        if not aceita.any():
            break
        mask[wy0:wy1, wx0:wx1] = sub_mask | aceita
        n_pix += int(aceita.sum())
        if n_pix > max_pixels:
            return np.zeros_like(mask, bool)
        ay, ax = np.where(aceita)
        y0, y1 = min(y0, wy0 + ay.min()), max(y1, wy0 + ay.max() + 1)
        x0, x1 = min(x0, wx0 + ax.min()), max(x1, wx0 + ax.max() + 1)
    return mask


def find_all_hotspots(gamma: np.ndarray) -> list:
    """Todos os hotspots (nao so o maior): acha o pico, cresce, remove a regiao e repete.
    Trabalha so no retangulo que contem os pixels nao nulos (a ROI): mesmo resultado, muito
    mais rapido quando ha centenas de picos (aletas, cabos)."""
    ys, xs = np.nonzero(gamma)
    if ys.size == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 2, xs.min(), xs.max() + 2
    y0, x0 = max(0, y0 - 1), max(0, x0 - 1)
    sub = _find_all_hotspots(gamma[y0:y1, x0:x1], gamma.size)
    out = []
    for h in sub:
        full = np.zeros(gamma.shape, bool)
        full[y0:y1, x0:x1] = h.mask
        out.append(Hotspot(full, h.n_pixels, h.t_hot, (h.centroid[0] + x0, h.centroid[1] + y0)))
    return out


def _find_all_hotspots(gamma: np.ndarray, total_px: int) -> list:
    working = gamma.copy()
    hotspots = []
    while working.max() > 0:
        seed = hottest_pixels_mask(working)
        region = grow_hotspots(working, seed, total_px=total_px)
        n = int(region.sum())
        if n >= MIN_HOTSPOT_PIXELS:
            ys, xs = np.where(region)
            hotspots.append(Hotspot(region, n, float(gamma[region].mean()),
                                    (float(xs.mean()), float(ys.mean()))))
        # se o crescimento abortou, remove ao menos a semente (evita loop infinito)
        working = np.where(region if region.any() else seed, 0, working)
    hotspots.sort(key=lambda h: h.n_pixels, reverse=True)
    return hotspots


def reference_stats(alpha: np.ndarray, excluir: np.ndarray) -> tuple:
    """(media, desvio) do fundo = pixels fora de `excluir`. Usa a imagem ORIGINAL (nao gamma):
    em objeto de borda abrupta a faixa morna do artigo fica vazia e T_ref degeneraria."""
    vals = alpha[~excluir]
    return (float(vals.mean()), float(vals.std())) if vals.size else (0.0, 0.0)


def run_itada_array(alpha: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> ITADAResult:
    """alpha: quadro termico float32 2D (maior = mais quente). roi_mask: regiao do componente."""
    alpha = alpha.astype(np.float32)
    hotspots = find_all_hotspots(segment(alpha, roi_mask))
    if not hotspots:
        pool = alpha[roi_mask] if roi_mask is not None else alpha
        return ITADAResult([], float(pool.mean()), 0.0)

    todos = np.zeros(alpha.shape, bool)
    for h in hotspots:
        todos |= h.mask
    t_ref, t_std = reference_stats(alpha, todos if roi_mask is None else (todos | ~roi_mask))
    limiar = max(MIN_CONTRAST_FLOOR, MIN_CONTRAST_SIGMAS * t_std)
    hotspots = [h for h in hotspots if h.t_hot - t_ref >= limiar]
    return ITADAResult(hotspots, t_ref, hotspots[0].t_hot - t_ref if hotspots else 0.0)


def load_thermal_image(path: str) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def run_itada(path: str, roi_mask: Optional[np.ndarray] = None) -> ITADAResult:
    return run_itada_array(load_thermal_image(path), roi_mask)


def draw_overlay(alpha: np.ndarray, hotspots: list, out_path: str) -> None:
    """Imagem com os hotspots contornados em vermelho e a area em px."""
    vis = cv2.cvtColor(cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    for h in hotspots:
        cnts, _ = cv2.findContours(h.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
        cv2.putText(vis, f"{h.n_pixels}px", (int(h.centroid[0]) + 5, int(h.centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    imwrite_unicode(out_path, vis)


def refinar_roi_otsu(gray: np.ndarray, roi_mask: np.ndarray, moldura_px: int = 3,
                     suavizar: float = 1.0, min_fracao: float = 0.05) -> np.ndarray:
    """Delimita o COMPONENTE dentro da caixa projetada (Zaripova et al., 2019): gaussiana ->
    Otsu -> objeto. Qual das duas classes e o componente e decidido pela MOLDURA da caixa
    (quase so fundo): objeto = a classe que mais difere dela, entao funciona com componente
    mais quente OU mais frio que o fundo. Depois: maior regiao conectada com buracos
    preenchidos (pontos quentes dentro do corpo ficam na ROI). Se nao conseguir, devolve a caixa."""
    roi_mask = roi_mask.astype(bool)
    if roi_mask.sum() < 50:
        return roi_mask
    gs = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), suavizar) if suavizar > 0 else gray.astype(np.float32)
    v = np.clip(gs[roi_mask], 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alto, baixo = roi_mask & (gs > t), roi_mask & (gs <= t)
    miolo = cv2.erode(roi_mask.astype(np.uint8), np.ones((2 * moldura_px + 1,) * 2, np.uint8)).astype(bool)
    moldura = roi_mask & ~miolo
    if alto.sum() < 10 or baixo.sum() < 10 or moldura.sum() < 10:
        return roi_mask
    fundo = float(np.median(gs[moldura]))
    obj = alto if abs(float(np.median(gs[alto])) - fundo) > abs(float(np.median(gs[baixo])) - fundo) else baixo
    u = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    if n <= 1:
        return roi_mask
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheio = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(cheio, cnts, -1, 1, -1)
    out = cheio.astype(bool) & roi_mask
    return out if out.sum() >= min_fracao * roi_mask.sum() else roi_mask


def filtrar_hotspots_fundo(gray: np.ndarray, roi: np.ndarray, hotspots: list, anel: int = 8,
                           margem: float = 10.0) -> list:
    """Descarta hotspots que NAO passam do fundo vizinho. Com componente mais FRIO que o fundo
    (isolador escuro), a borda borrada entre os dois vira 'hotspot' com valor no maximo igual ao
    do fundo; um ponto quente real passa do fundo. Compara o PICO do hotspot com a mediana de um
    anel de `anel` px logo fora da ROI + `margem` (niveis de cinza).
    Custo: perde defeito que, mesmo aquecido, continua mais frio que o fundo."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * anel + 1,) * 2)
    ring = cv2.dilate(roi.astype(np.uint8), k).astype(bool) & ~roi
    if ring.sum() < 20:
        return list(hotspots)
    ref = float(np.median(gray[ring]))
    return [h for h in hotspots if float(gray[h.mask].max()) > ref + margem]


def termico_colorido(th_bgr: Optional[np.ndarray], limite: float = 30.0) -> bool:
    """True se o quadro termico tem paleta de COR (em white-hot os 3 canais sao iguais).
    Serve para avisar se a C12 ficou configurada errado. Limite 30: a compressao de video
    cria ~5-10 de diferenca entre canais num quadro cinza; uma paleta colorida da > 100."""
    if th_bgr is None or th_bgr.ndim < 3:
        return False
    f = th_bgr[::2, ::2].astype(np.int16)
    return bool(np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean() > limite)


def avaliar_hotspot(thermal_bgr: np.ndarray, box_rgb: tuple, H: np.ndarray,
                    roi_margin: float = 0.15, roi_refino: bool = False,
                    filtro_fundo: Optional[float] = None,
                    min_roi_px: int = 8, min_inside: float = 0.5) -> dict:
    """Projeta a caixa do YOLO no termico (homografia), expande por `roi_margin` (fracao),
    opcionalmente refina por Otsu e roda o ITADA so nessa regiao.
    filtro_fundo: None = desligado; um numero = margem (niveis de cinza) do filtrar_hotspots_fundo.
    -> dict: ok, motivo, quad, caixa, roi_mask, hotspot, delta_t, hotspots, centroides_rgb."""
    th_h, th_w = thermal_bgr.shape[:2]
    tb = transform_bbox(box_rgb, H)
    x1, y1, x2, y2 = tb["bbox_axis_aligned"]
    mw, mh = (x2 - x1) * roi_margin, (y2 - y1) * roi_margin
    x1, y1, x2, y2 = x1 - mw, y1 - mh, x2 + mw, y2 + mh
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(th_w, x2), min(th_h, y2)
    area_dentro = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

    res = {"ok": False, "motivo": None, "quad": tb["quad"], "caixa": (cx1, cy1, cx2, cy2),
           "roi_mask": None, "hotspot": False, "delta_t": None, "hotspots": [], "centroides_rgb": []}
    if area_dentro / max(1e-6, (x2 - x1) * (y2 - y1)) < min_inside:
        return {**res, "motivo": "fora_do_campo_termico"}
    if (cx2 - cx1) < min_roi_px or (cy2 - cy1) < min_roi_px:
        return {**res, "motivo": "roi_muito_pequena"}

    g = (cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY) if thermal_bgr.ndim == 3
         else thermal_bgr).astype(np.float32)
    roi = bbox_to_roi_mask((cx1, cy1, cx2, cy2), g.shape)
    if roi_refino:
        roi = refinar_roi_otsu(g, roi)
    if roi.sum() < 20:
        return {**res, "motivo": "roi_muito_pequena"}
    res["roi_mask"] = roi

    try:
        it = run_itada_array(g, roi)
    except Exception as e:   # ROI degenerada etc. -- nao derruba a inspecao
        return {**res, "motivo": f"itada_erro:{type(e).__name__}"}
    if filtro_fundo is not None:
        it.hotspots = filtrar_hotspots_fundo(g, roi, it.hotspots, margem=filtro_fundo)

    cents = []
    if it.hotspots:
        pts = np.array([[h.centroid] for h in it.hotspots], dtype=np.float32)
        cents = [tuple(map(float, p)) for p in cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)]
    return {**res, "ok": True, "hotspot": bool(it.hotspots),
            "delta_t": (it.hotspots[0].t_hot - it.t_ref) if it.hotspots else None,
            "hotspots": it.hotspots, "centroides_rgb": cents}


# =============================================================================
# PARTE 4 -- HOMOGRAFIA RGB -> termico
# =============================================================================
# A C12 tem lentes RGB (100 x 52 graus, 1280x720) e termica (24.8 x 18.7 graus, 384x288)
# separadas. A homografia so e exata para cena plana ou centro optico comum: ha paralaxe
# residual, aceitavel a poucos metros ou mais, ruim em cena muito proxima.
#
# CALIBRACAO (uma vez; as lentes sao fixas entre si):
#   1. pares RGB + termico SIMULTANEOS de uma fonte de calor pequena (ferro de solda,
#      resistor, lampada), visivel nos dois;
#   2. 8-10 posicoes bem espalhadas pelo quadro (cantos, centro, bordas);
#   3. anote o pixel da fonte em cada imagem num JSON:
#        [{"rgb": [120, 340], "thermal": [45, 130]}, ...]
#   4. python modelos.py calibrar pontos.json homografia_c12.npy
# A calibracao deve ser feita na MESMA resolucao dos streams usados na inspecao.

def compute_homography(points_rgb, points_thermal, ransac_threshold: float = 3.0):
    """-> (H, mask_inliers). RANSAC tolera erros de anotacao manual."""
    a = np.asarray(points_rgb, np.float32).reshape(-1, 1, 2)
    b = np.asarray(points_thermal, np.float32).reshape(-1, 1, 2)
    if len(a) < 4:
        raise ValueError(f"Precisa de >= 4 pontos (tem {len(a)}); recomendado 8-10 bem espalhados.")
    H, mask = cv2.findHomography(a, b, cv2.RANSAC, ransac_threshold)
    if H is None:
        raise RuntimeError("Homografia falhou: pontos colineares ou concentrados numa regiao pequena?")
    return H, mask


def transform_point(xy: tuple, H: np.ndarray) -> tuple:
    x, y = cv2.perspectiveTransform(np.array([[xy]], np.float32), H)[0, 0]
    return float(x), float(y)


def transform_bbox(bbox_xyxy: tuple, H: np.ndarray) -> dict:
    """Caixa RGB -> {'quad': 4 cantos no termico, 'bbox_axis_aligned': caixa envolvente}.
    A homografia pode inclinar o retangulo, por isso as duas formas."""
    x1, y1, x2, y2 = bbox_xyxy
    c = cv2.perspectiveTransform(np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float32),
                                 H).reshape(-1, 2)
    return {"quad": c.tolist(),
            "bbox_axis_aligned": (float(c[:, 0].min()), float(c[:, 1].min()),
                                  float(c[:, 0].max()), float(c[:, 1].max()))}


def bbox_to_roi_mask(bbox_xyxy: tuple, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    m = np.zeros((h, w), bool)
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


def calibrar(json_path: str, saida_npy: str) -> np.ndarray:
    """Calcula a homografia, mostra o erro de reprojecao por ponto e salva o .npy."""
    with open(json_path, encoding="utf-8") as f:
        pares = json.load(f)
    a = np.array([p["rgb"] for p in pares], np.float64)
    b = np.array([p["thermal"] for p in pares], np.float64)
    H, mask = compute_homography(a, b)
    erros = np.array([np.hypot(*(np.array(transform_point(p, H)) - q)) for p, q in zip(a, b)])
    inl = mask.ravel().astype(bool)
    print(f"{len(pares)} pontos, {int(inl.sum())} inliers (RANSAC)")
    print(f"{'#':>3s} {'rgb':>14s} {'termico':>14s} {'erro (px)':>10s}")
    for i, (p, q, e, ok) in enumerate(zip(a, b, erros, inl)):
        print(f"{i:3d} {f'({int(p[0])}, {int(p[1])})':>14s} {f'({int(q[0])}, {int(q[1])})':>14s} "
              f"{e:10.2f}{'' if ok else '  <- outlier (reveja este par)'}")
    media = float(erros[inl].mean()) if inl.any() else float("nan")
    print(f"\nErro medio de reprojecao (inliers): {media:.2f} px")
    if media > 3.0:
        print("[AVISO] erro medio > 3 px: a ROI vai sair deslocada. Refaca com mais pontos espalhados.")
    np.save(saida_npy, H)
    print(f"Homografia salva em: {saida_npy}")
    return H


# =============================================================================
# Linha de comando
# =============================================================================
def _cmd_itada(pasta: str, saida: Optional[str]):
    if saida:
        os.makedirs(saida, exist_ok=True)
    for f in sorted(os.listdir(pasta)):
        if not f.lower().endswith(EXT_IMAGEM):
            continue
        alpha = load_thermal_image(os.path.join(pasta, f))
        r = run_itada_array(alpha)
        print(f"{f:30s} hotspots={len(r.hotspots):3d}  delta_t={r.delta_t:6.1f}  t_ref={r.t_ref:6.1f}")
        if saida and r.hotspots:
            draw_overlay(alpha, r.hotspots, os.path.join(saida, f"overlay_{os.path.splitext(f)[0]}.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("yolo", help="treina o YOLO de deteccao dos componentes")
    a.add_argument("data_yaml")
    a.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m/l/x.pt (ou yolo11n.pt...)")
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--device", default="0", help="'0' = GPU 0, 'cpu' = sem GPU")
    a.add_argument("--project", default="runs_yolo")
    a.add_argument("--name", default="treino")

    a = sub.add_parser("validar", help="metricas por classe de um best.pt")
    a.add_argument("weights")
    a.add_argument("data_yaml")
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--device", default="0")
    a.add_argument("--saida", default="validacao_yolo")

    a = sub.add_parser("patchcore", help="treina o PatchCore de UMA classe do InsPLAD-fault")
    a.add_argument("pasta_classe", help="pasta da classe com train/good e test/<good|defeitos>")
    a.add_argument("--name", required=True, help="use o MESMO nome da classe no YOLO")
    a.add_argument("--batch-size", type=int, default=8)
    a.add_argument("--coreset", type=float, default=0.1)

    a = sub.add_parser("itada", help="roda o ITADA nas imagens termicas de uma pasta")
    a.add_argument("pasta")
    a.add_argument("--saida", default=None, help="pasta para salvar os overlays")

    a = sub.add_parser("calibrar", help="homografia RGB -> termico a partir do JSON de pontos")
    a.add_argument("pontos_json")
    a.add_argument("saida_npy")


    a = p.parse_args()
    if a.cmd == "yolo":
        treinar_yolo(a.data_yaml, a.model, a.epochs, a.imgsz, a.batch, a.device, a.project, a.name)
    elif a.cmd == "validar":
        validar_yolo(a.weights, a.data_yaml, a.imgsz, a.device, a.saida)
    elif a.cmd == "patchcore":
        treinar_patchcore(a.pasta_classe, a.name, a.batch_size, a.coreset)
    elif a.cmd == "itada":
        _cmd_itada(a.pasta, a.saida)
    else:
        calibrar(a.pontos_json, a.saida_npy)


if __name__ == "__main__":
    main()
