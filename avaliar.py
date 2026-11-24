"""
AVALIACAO dos detectores (substitui o avaliar.py antigo e o calibrate_threshold.py).

  anomalia  PatchCore de UMA classe do InsPLAD-fault: scores em test/good e em cada
            test/<tipo_de_defeito>, AUROC geral e por tipo, AUPR, heatmaps e LIMIAR
            SUGERIDO para o base_inspecao.py (--limiar CLASSE=VALOR).
  itada     ITADA dentro das caixas ROTACIONADAS de um dataset termico (ex.: Insulator
            Infrared Rotated Object Detection Dataset, Zenodo), com a MESMA ROI do
            base_inspecao.py (caixa alinhada + margem, refino Otsu opcional). Com um
            esperados.csv (onde voce colou os hotspots), mede deteccao e falso alarme.

As imagens termicas ja devem chegar prontas (white-hot / cinza, hotspots ja injetados):
este script NAO altera imagem. Ele processa so as imagens que estiverem na pasta, entao
basta copiar as que forem aproveitadas.

USO:
    python avaliar.py anomalia patchcore_glass/.../model.ckpt insplad_fault/glass-insulator saida_anomalia/
    python avaliar.py itada pasta_termica/ saida_itada/ [--esperados esperados.csv] [--roi-refino]

ESTRUTURA esperada no `itada`:
    pasta_termica/images/*.png|jpg  +  rotulos com o mesmo nome em rlabels/, labels/ ou
    annotations/ (ou ao lado da imagem), em um destes formatos:
      .txt  YOLO-OBB: "classe x1 y1 x2 y2 x3 y3 x4 y4" normalizado 0-1 (nomes em classes.txt)
      .xml  roLabelImg: <object><name>..</name><robndbox><cx><cy><w><h><angle(rad)></robndbox>;
            <bndbox> comum tambem e aceito.

esperados.csv (opcional), um hotspot colado por linha, em pixels da imagem:
    imagem,x,y[,contraste][,raio][,...]
  - imagem SEM linha no CSV = imagem sem hotspot (conta para o falso alarme);
  - colunas extras (contraste, raio, ...) viram uma tabela de deteccao por valor.
"""
import os
import csv
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import cv2

import modelos
from modelos import EXT_IMAGEM


# =============================================================================
# Metricas comuns
# =============================================================================
def auroc(pos, neg) -> float:
    """Probabilidade de um positivo ter score maior que um negativo (Mann-Whitney)."""
    if not len(pos) or not len(neg):
        return float("nan")
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def aupr(pos, neg) -> float:
    """Average precision com o defeito como positivo (importa quando ha muito mais normais)."""
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])[np.argsort(-s, kind="stable")]
    tp = np.cumsum(y)
    return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / len(pos))


def sugerir_limiar(pos, neg, fpr_max: float = 0.10) -> dict:
    """Regra de alarme: score >= limiar. -> {'f1': {...}, 'recall': {...} | None}
    f1     = limiar que maximiza o F1
    recall = MAIOR recall com falso positivo <= fpr_max (prioriza nao deixar passar falha).
    O base_inspecao ainda exige --min-frames seguidos, entao ~10% por imagem costuma ir bem."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)

    def met(t):
        tp, fp = int((pos >= t).sum()), int((neg >= t).sum())
        rec, prec = tp / len(pos), (tp / (tp + fp) if tp + fp else 0.0)
        return {"limiar": float(t), "recall": rec, "fpr": fp / len(neg), "precisao": prec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}

    todas = [met(t) for t in np.unique(np.concatenate([pos, neg]))]
    ok = [m for m in todas if m["fpr"] <= fpr_max]
    return {"f1": max(todas, key=lambda m: (m["f1"], m["limiar"])),
            "recall": max(ok, key=lambda m: (m["recall"], m["limiar"])) if ok else None,
            "fpr_max": fpr_max}


def _imagens(pasta: str) -> list:
    return sorted(f for f in os.listdir(pasta) if f.lower().endswith(EXT_IMAGEM)) if os.path.isdir(pasta) else []


# =============================================================================
# ANOMALIA (PatchCore, InsPLAD-fault)
# =============================================================================
def avaliar_pasta_patchcore(pc, pasta: str, out_dir, prefixo: str) -> list:
    """Score de cada imagem (o recorte do InsPLAD-fault inteiro, como no treino) e heatmap."""
    out = []
    for f in _imagens(pasta):
        img = modelos.imread_unicode(os.path.join(pasta, f))
        if img is None:
            continue
        score, amap = modelos.inferir_patchcore(pc, img)
        out.append({"arquivo": f, "tipo": prefixo, "score": score})
        if out_dir and amap is not None:
            a8 = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cor = cv2.resize(cv2.applyColorMap(a8, cv2.COLORMAP_JET), (img.shape[1], img.shape[0]))
            modelos.imwrite_unicode(os.path.join(out_dir, "heatmaps", f"{prefixo}__{f}"),
                                    cv2.addWeighted(img, 0.6, cor, 0.4, 0))
    return out


def avaliar_anomalia(ckpt: str, pasta_classe: str, out_dir: str, heatmaps: bool = True,
                     fpr_max: float = 0.10) -> dict:
    os.makedirs(os.path.join(out_dir, "heatmaps"), exist_ok=True)
    pc = modelos.carregar_modelo_patchcore(ckpt)
    test = os.path.join(pasta_classe, "test")
    tipos = modelos.pastas_defeito(pasta_classe)
    hd = out_dir if heatmaps else None

    print("Avaliando test/good ...")
    normais = avaliar_pasta_patchcore(pc, os.path.join(test, "good"), hd, "good")
    defeitos = []
    for t in tipos:
        print(f"Avaliando test/{t} ...")
        defeitos += avaliar_pasta_patchcore(pc, os.path.join(test, t), hd, t)

    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "tipo", "score"])
        w.writeheader()
        w.writerows(normais + defeitos)

    sn = [s["score"] for s in normais]
    sd = [s["score"] for s in defeitos]
    if not sn:
        print(f"[ERRO] nenhuma imagem em {test}/good")
        return {}
    print(f"\n=== RESULTADO ({os.path.basename(os.path.normpath(pasta_classe))}) ===")
    print(f"normais: n={len(sn)}  score medio {np.mean(sn):.4f}")
    if not sd:
        print("Sem pastas de defeito em test/: AUROC e limiar nao calculaveis.")
        return {"normais": normais}
    print(f"defeito: n={len(sd)}  score medio {np.mean(sd):.4f}")
    print(f"AUROC (imagem): {auroc(sd, sn):.4f}   (0.5 = chute, 1.0 = perfeito)")
    print(f"AUPR (defeito positivo): {aupr(sd, sn):.4f}   (chute = {len(sd) / (len(sd) + len(sn)):.3f})")

    print("\n=== AUROC POR TIPO DE DEFEITO (vs. todos os normais) ===")
    for t in tipos:
        st = [s["score"] for s in defeitos if s["tipo"] == t]
        print(f"  {t:24s} n={len(st):4d}  AUROC={auroc(st, sn):.4f}")

    sug = sugerir_limiar(sd, sn, fpr_max)
    m = sug["f1"]
    print("\n=== LIMIAR SUGERIDO (base_inspecao.py --limiar CLASSE=VALOR) ===")
    print(f"Melhor F1: {m['limiar']:.4f}  (recall {m['recall']:.2f}, FPR {m['fpr']:.2f}, F1 {m['f1']:.2f})")
    m = sug["recall"]
    if m:
        print(f"Max recall com FPR <= {fpr_max:.0%}: {m['limiar']:.4f}  (recall {m['recall']:.2f}, "
              f"FPR {m['fpr']:.2f})  <- recomendado (prioriza recall)")
    else:
        print(f"Nenhum limiar com FPR <= {fpr_max:.0%}: normais e defeitos se sobrepoem.")
    print("Obs.: o teste e pequeno; trate como ponto de partida e confirme no video real.")
    print(f"scores.csv{' e heatmaps' if heatmaps else ''} em: {out_dir}")
    return {"auroc": auroc(sd, sn), "normais": normais, "defeitos": defeitos, "limiar": sug}


# =============================================================================
# ITADA nas caixas rotacionadas (dataset termico)
# =============================================================================
def ler_caixas_xml(xml_path: str) -> list:
    """-> [(classe, poligono 4x2 float)] de <robndbox> (roLabelImg) ou <bndbox> (VOC)."""
    out = []
    for o in ET.parse(xml_path).getroot().findall("object"):
        nome = o.findtext("name") or o.findtext("n") or "?"
        r, b = o.find("robndbox"), o.find("bndbox")
        if r is not None:
            cx, cy, w, h, ang = (float(r.findtext(k)) for k in ("cx", "cy", "w", "h", "angle"))
            out.append((nome, cv2.boxPoints(((cx, cy), (w, h), np.degrees(ang)))))
        elif b is not None:
            x0, y0, x1, y1 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            out.append((nome, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)))
    return out


def ler_caixas_obb_txt(txt_path: str, w: int, h: int, nomes: list) -> list:
    """YOLO-OBB normalizado -> [(classe, poligono 4x2 em px)]."""
    out = []
    for linha in open(txt_path):
        v = linha.split()
        if len(v) >= 9:
            cid = int(float(v[0]))
            nome = nomes[cid] if cid < len(nomes) else str(cid)
            out.append((nome, (np.array(v[1:9], float).reshape(4, 2) * [w, h]).astype(np.float32)))
    return out


def _achar_rotulo(raiz: str, pasta_img: str, nome_img: str):
    base = os.path.splitext(nome_img)[0]
    for d in [os.path.join(raiz, s) for s in ("rlabels", "labels", "annotations")] + [pasta_img]:
        for ext in (".xml", ".txt"):
            if os.path.exists(os.path.join(d, base + ext)):
                return os.path.join(d, base + ext)
    return None


def ler_esperados(path: str):
    """-> ({imagem: [(x, y, {extras})]}, [nomes das colunas extras])"""
    pts, extras = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        extras = [c for c in r.fieldnames if c not in ("imagem", "x", "y")]
        for row in r:
            pts.setdefault(row["imagem"].strip(), []).append(
                (float(row["x"]), float(row["y"]), {c: row[c] for c in extras}))
    return pts, extras


def _mascara_poligono(shape, pol) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pol).astype(np.int32)], 1)
    return m.astype(bool)


def _rodar_caixa(img, pol, modo_roi, roi_margin, roi_refino, filtro_fundo=None) -> dict:
    """Mesma ROI do base_inspecao: 'caixa' = retangulo alinhado que envolve a caixa girada,
    expandido por roi_margin, passado por modelos.avaliar_hotspot (homografia identidade).
    'rotacionada' = o poligono exato da anotacao (mais justo; nao existe no pipeline real)."""
    if modo_roi == "caixa":
        x0, y0 = pol.min(0)
        x1, y1 = pol.max(0)
        return modelos.avaliar_hotspot(img, (x0, y0, x1, y1), np.eye(3), roi_margin, roi_refino, filtro_fundo)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    roi = _mascara_poligono(g.shape, pol)
    if roi_refino:
        roi = modelos.refinar_roi_otsu(g, roi)
    if roi.sum() < 20:
        return {"ok": False, "motivo": "roi_muito_pequena", "hotspots": [], "delta_t": None, "roi_mask": roi}
    r = modelos.run_itada_array(g, roi)
    if filtro_fundo is not None:
        r.hotspots = modelos.filtrar_hotspots_fundo(g, roi, r.hotspots, margem=filtro_fundo)
    return {"ok": True, "motivo": None, "hotspots": r.hotspots, "delta_t": r.delta_t if r.hotspots else None,
            "roi_mask": roi}


def _taxa(a, n) -> str:
    return f"{a:4d}/{n:<4d} {a / n:6.1%}" if n else f"{a:4d}/{n:<4d}     - "


def avaliar_itada(pasta: str, out_dir: str, esperados_csv=None, classe=None, modo_roi="caixa",
                  roi_margin: float = 0.15, roi_refino: bool = False, tol: float = 10.0,
                  min_pixels=None, overlays: bool = True, filtro_fundo=None) -> list:
    os.makedirs(out_dir, exist_ok=True)
    if min_pixels is not None:
        modelos.MIN_HOTSPOT_PIXELS = int(min_pixels)
    pasta_img = os.path.join(pasta, "images") if os.path.isdir(os.path.join(pasta, "images")) else pasta
    esp, extras = ler_esperados(esperados_csv) if esperados_csv else ({}, [])
    p_cls = os.path.join(pasta, "classes.txt")
    nomes = [l.strip() for l in open(p_cls) if l.strip()] if os.path.exists(p_cls) else []
    tem_gt = esperados_csv is not None

    linhas, sem_xml, avisou_cor, achados = [], 0, False, []   # achados: (extras, achou_no_lugar)
    for f in _imagens(pasta_img):
        rot = _achar_rotulo(pasta, pasta_img, f)
        if rot is None:
            sem_xml += 1
            continue
        img = modelos.imread_unicode(os.path.join(pasta_img, f))
        if img is None:
            continue
        if not avisou_cor and modelos.termico_colorido(img):
            avisou_cor = True
            print(f"[AVISO] {f} tem paleta de COR: o ITADA espera white-hot (cinza, quente = claro).")
        pts_img = esp.get(f, [])
        usados = set()
        vis = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        caixas = (ler_caixas_obb_txt(rot, img.shape[1], img.shape[0], nomes) if rot.endswith(".txt")
                  else ler_caixas_xml(rot))
        for cid, (nome, pol) in enumerate(caixas):
            if classe and nome != classe:
                continue
            res = _rodar_caixa(img, pol, modo_roi, roi_margin, roi_refino, filtro_fundo)
            dentro = _mascara_poligono(img.shape, pol)
            # hotspots esperados DESTA caixa: os pontos colados dentro do poligono anotado
            meus = [(i, p) for i, p in enumerate(pts_img)
                    if 0 <= int(p[1]) < img.shape[0] and 0 <= int(p[0]) < img.shape[1] and dentro[int(p[1]), int(p[0])]]
            cents = [h.centroid for h in res["hotspots"]]
            perto = lambda c: any(np.hypot(c[0] - p[0], c[1] - p[1]) <= tol for _, p in meus)
            achou = [any(np.hypot(c[0] - p[0], c[1] - p[1]) <= tol for c in cents) for _, p in meus]
            for (i, p), a in zip(meus, achou):
                usados.add(i)
                achados.append((p[2], a))
            linhas.append({
                "imagem": f, "caixa": cid, "classe": nome, "ok": res["ok"], "motivo": res["motivo"] or "",
                "n_hotspots": len(cents),
                "delta_t": "" if res["delta_t"] is None else round(float(res["delta_t"]), 2),
                "gt_hotspots": len(meus) if tem_gt else "",
                "achados_no_lugar": sum(achou) if tem_gt else "",
                "extras": sum(not perto(c) for c in cents) if tem_gt else ""})

            if overlays:
                cor = (0, 0, 255) if cents else (0, 200, 0)
                cv2.polylines(vis, [np.round(pol).astype(np.int32)], True, cor, 1)
                if res.get("roi_mask") is not None:
                    c, _ = cv2.findContours(res["roi_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, c, -1, (255, 200, 0), 1)
                for h in res["hotspots"]:
                    c, _ = cv2.findContours(h.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, c, -1, (0, 0, 255), 1)
        if overlays:
            for i, (x, y, _) in enumerate(pts_img):
                cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 255) if i in usados else (255, 0, 255),
                               cv2.MARKER_CROSS, 10, 1)
            os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
            modelos.imwrite_unicode(os.path.join(out_dir, "overlays", os.path.splitext(f)[0] + ".png"), vis)
        fora = [p for i, p in enumerate(pts_img) if i not in usados]
        if fora:
            print(f"[AVISO] {f}: {len(fora)} ponto(s) do esperados.csv fora de qualquer caixa (ignorados)")

    if sem_xml:
        print(f"[AVISO] {sem_xml} imagem(ns) sem rotulo foram ignoradas.")
    if not linhas:
        print("Nenhuma caixa avaliada. Confira a pasta (images/ + rotulos) e --classe.")
        return []
    with open(os.path.join(out_dir, "itada_caixas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    validas = [l for l in linhas if l["ok"]]
    print(f"\n{len({l['imagem'] for l in linhas})} imagens, {len(linhas)} caixas "
          f"({len(linhas) - len(validas)} com ROI invalida); ROI = {modo_roi}"
          f"{' + refino Otsu' if roi_refino else ''}{f' + filtro de fundo {filtro_fundo:g}' if filtro_fundo is not None else ''}, margem {roi_margin:g}, "
          f"MIN_HOTSPOT_PIXELS = {modelos.MIN_HOTSPOT_PIXELS}")
    if not tem_gt:
        com = [l for l in validas if l["n_hotspots"]]
        print(f"Caixas com hotspot: {_taxa(len(com), len(validas))}")
        if com:
            print(f"delta_t (niveis de cinza) nas caixas com hotspot: mediana "
                  f"{np.median([l['delta_t'] for l in com]):.1f}, faixa {min(l['delta_t'] for l in com):.1f}"
                  f" - {max(l['delta_t'] for l in com):.1f}")
        print("Sem --esperados: nao ha como separar acerto de falso alarme.")
    else:
        pos = [l for l in validas if l["gt_hotspots"]]
        neg = [l for l in validas if not l["gt_hotspots"]]
        n_pts = sum(l["gt_hotspots"] for l in pos)
        print("\n=== POR CAIXA (componente) ===")
        print(f"caixas COM hotspot colado, sinalizadas:   {_taxa(sum(l['n_hotspots'] > 0 for l in pos), len(pos))}")
        print(f"caixas SEM hotspot, sinalizadas (alarme): {_taxa(sum(l['n_hotspots'] > 0 for l in neg), len(neg))}")
        print("\n=== POR HOTSPOT COLADO ===")
        print(f"achados no lugar (<= {tol:g} px):            {_taxa(sum(l['achados_no_lugar'] for l in pos), n_pts)}")
        print(f"hotspots extras (longe de qualquer ponto): {sum(l['extras'] for l in validas)}")
        for c in extras:
            grupos = {}
            for ex, a in achados:
                grupos.setdefault(ex.get(c, ""), []).append(a)
            print(f"\n=== DETECCAO NO LUGAR POR '{c}' ===")
            chave = lambda v: (0, float(v)) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else (1, v)
            for v in sorted(grupos, key=chave):
                print(f"  {c}={v:>8s}: {_taxa(sum(grupos[v]), len(grupos[v]))}")
    print(f"\nResultados em {out_dir}/itada_caixas.csv"
          f"{' e overlays/ (caixa anotada; ciano = ROI; vermelho = hotspot; cruz amarela = ponto achado; magenta = fora de caixa)' if overlays else ''}")
    print("Leitura: delta_t e em NIVEIS DE CINZA, nao graus C.")
    return linhas


# =============================================================================
# Linha de comando
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("anomalia", help="PatchCore de uma classe do InsPLAD-fault")
    a.add_argument("checkpoint")
    a.add_argument("pasta_classe", help="pasta da classe com test/good e test/<tipos de defeito>")
    a.add_argument("out_dir")
    a.add_argument("--sem-heatmap", action="store_true")
    a.add_argument("--fpr-max", type=float, default=0.10, help="falso positivo maximo para o limiar recomendado")

    a = sub.add_parser("itada", help="ITADA nas caixas rotacionadas de um dataset termico")
    a.add_argument("pasta", help="pasta com images/ e os rotulos (labels/*.txt YOLO-OBB ou rlabels/*.xml)")
    a.add_argument("out_dir")
    a.add_argument("--esperados", default=None, help="esperados.csv: imagem,x,y[,contraste,raio,...]")
    a.add_argument("--classe", default=None, help="so caixas desta classe (padrao: todas)")
    a.add_argument("--roi", choices=["caixa", "rotacionada"], default="caixa",
                   help="caixa = igual ao base_inspecao (retangulo + margem); rotacionada = poligono exato")
    a.add_argument("--roi-margin", type=float, default=0.15)
    a.add_argument("--roi-refino", action="store_true", help="refino Otsu do componente dentro da ROI")
    a.add_argument("--tol", type=float, default=10.0, help="distancia (px) para contar o hotspot no lugar")
    a.add_argument("--min-pixels", type=int, default=None,
                   help="area minima do hotspot (padrao 5). Em 640x640 as aletas geram pontinhos: experimente 30")
    a.add_argument("--filtro-fundo", type=float, default=None, metavar="MARGEM",
                   help="descarta hotspot que nao passa do fundo ao redor da ROI + MARGEM (igual ao base_inspecao)")
    a.add_argument("--sem-overlay", action="store_true")
    a = p.parse_args()

    if a.cmd == "anomalia":
        avaliar_anomalia(a.checkpoint, a.pasta_classe, a.out_dir, not a.sem_heatmap, a.fpr_max)
    else:
        avaliar_itada(a.pasta, a.out_dir, a.esperados, a.classe, a.roi, a.roi_margin, a.roi_refino,
                      a.tol, a.min_pixels, not a.sem_overlay, a.filtro_fundo)


if __name__ == "__main__":
    main()
