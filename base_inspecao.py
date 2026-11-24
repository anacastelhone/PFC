"""
BASE DE INSPECAO AO VIVO (substitui pipeline_video.py e a parte de stream do Imagens_GPS.py)
==========================================================================================
Roda no computador da BASE. Le os streams RGB e termico da Skydroid C12 e o GPS do LoRa AO
VIVO, processa os quadros numa FILA e salva cada falha assim que ela e confirmada:

  YOLO        detecta o componente (isolador de vidro/polimerico, vari-grip, espacador, stockbridge)
  PatchCore   anomalia fisica no recorte RGB (so nas classes que tem checkpoint)
  ITADA       hotspot no termico, dentro da caixa do componente projetada pela homografia

COMO FUNCIONA
  threads leitoras  RGB, termico e GPS em paralelo, sempre com o dado mais recente
                    (o termico guarda uns segundos de historico para casar por tempo com o RGB)
  amostrador        a cada 1/--fps-proc s tira um INSTANTANEO: quadro RGB + termico mais proximo
                    no tempo + GPS naquele instante, e poe na fila. Casar na captura (e nao no
                    processamento) mantem o GPS certo mesmo que o processamento atrase.
  fila limitada     se o processamento nao acompanha, o instantaneo mais ANTIGO e descartado
  processador       detecta, rastreia (IoU), calcula anomalia e hotspot, confirma a falha em
                    --min-frames instantaneos e salva o EVENTO (1 por componente e tipo de falha)

  Caixa VERDE = componente sem falha confirmada; VERMELHA = falha confirmada.

SAIDAS (em --out-dir, padrao inspecao_AAAAMMDD_HHMMSS)
  eventos.csv / eventos.json     atualizados a cada evento novo
  eventos/evt_0001_<classe>_<anomalia|hotspot>/
      rgb.jpg  rgb_anotado.jpg  termica.jpg  termica_anotada.jpg  evento.json
  video_anotado.mp4              quadros processados, com caixas
  bruto/ (com --gravar-bruto)    rgb.mp4, termico.mp4, gps.csv dos quadros processados,
                                 para reprocessar depois com --reprocessar

USO
  # inspecao ao vivo
  python base_inspecao.py --yolo-weights best.pt \
      --patchcore glass-insulator=pc_vidro.ckpt --patchcore vari-grip=pc_varigrip.ckpt \
      --rgb-url rtsp://192.168.144.108:554/stream=1 --thermal-url rtsp://192.168.144.108:554/stream=0 \
      --homography homografia_c12.npy --gps-port COM5 --roi-refino --gravar-bruto

  # reprocessar uma gravacao bruta com outros parametros (sem tempo real)
  python base_inspecao.py --yolo-weights best.pt --patchcore ... --homography homografia_c12.npy \
      --reprocessar inspecao_20261120_101500/bruto --out-dir reprocessado

  # descobrir as URLs dos streams da C12 (nao carrega modelos)
  python base_inspecao.py --testar-streams [url1 url2 ...]

  Ctrl+C encerra com seguranca (os eventos ja estao salvos). --rgb-url tambem aceita um arquivo
  de video (reproduzido em tempo real, para testar sem o robo) ou um indice de webcam (0).
  A C12 deve estar com o termico em WHITE-HOT (quente = claro).
"""
import os
import re
import csv
import json
import time
import queue
import argparse
import threading
import collections
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

import numpy as np
import cv2

import modelos
from modelos import Detection

VERDE, VERMELHO, BRANCO, PRETO = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)
FALHA_ANOMALIA = "anomalia detectada"
FALHA_HOTSPOT = "hotspot detectado"
TAG_FALHA = {FALHA_ANOMALIA: "anomalia", FALHA_HOTSPOT: "hotspot"}


# =============================================================================
# GPS ao vivo
# =============================================================================
_RE_GPS = re.compile(r"LAT:\s*(-?\d+(?:\.\d+)?)\s*,\s*LON:\s*(-?\d+(?:\.\d+)?)\s*,\s*SAT:\s*(\d+)")


def parse_gps(linha: str):
    """'LAT:-19.74723,LON:-47.93811,SAT:8' (formato do LoRa) -> (lat, lon, sat) ou None."""
    m = _RE_GPS.search(linha or "")
    return (float(m.group(1)), float(m.group(2)), int(m.group(3))) if m else None


def linhas_serial(porta: str, baud: int = 115200) -> Iterable[str]:
    """Linhas da porta serial; reconecta sozinho se a porta cair."""
    import serial   # pyserial
    avisou = False
    while True:
        try:
            ser = serial.Serial(porta, baud, timeout=1)
            print(f"[GPS] serial conectada ({porta})")
            avisou = False
            while True:
                bruto = ser.readline()
                if bruto:
                    yield bruto.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            if not avisou:
                print(f"[GPS] serial indisponivel ({e}); tentando de novo...")
                avisou = True
            time.sleep(2.0)


class GPSAoVivo(threading.Thread):
    """Guarda a ultima posicao valida. Posicao 0,0 (antes do fix) e ignorada.
    status: 'ok' | 'sem_fix' (nenhuma posicao, ou mais velha que max_idade_s) | 'sem_log' (sem GPS)."""

    def __init__(self, linhas: Optional[Iterable[str]] = None, max_idade_s: float = 5.0):
        super().__init__(daemon=True)
        self.linhas, self.max_idade_s = linhas, max_idade_s
        self._lock = threading.Lock()
        self._ult = None          # (lat, lon, sat, t_recebido)

    def run(self):
        if self.linhas is None:
            return
        for linha in self.linhas:
            r = parse_gps(linha)
            if r is None or (r[0] == 0.0 and r[1] == 0.0):
                continue
            with self._lock:
                self._ult = (r[0], r[1], r[2], time.time())

    def instantaneo(self, ts: float) -> dict:
        vazio = {"lat": None, "lon": None, "sat": None}
        if self.linhas is None:
            return {**vazio, "status": "sem_log"}
        with self._lock:
            u = self._ult
        if u is None or abs(ts - u[3]) > self.max_idade_s:
            return {**vazio, "status": "sem_fix"}
        return {"lat": u[0], "lon": u[1], "sat": u[2], "status": "ok"}


# =============================================================================
# Streams ao vivo
# =============================================================================
class FonteViva(threading.Thread):
    """Le um stream (RTSP, webcam ou ARQUIVO) numa thread e guarda o quadro mais recente.
    Reconecta se cair. Arquivo: reproduz em tempo real e marca `terminou` no fim.
    historico_s > 0 guarda os quadros dos ultimos segundos (termico, para casar por tempo)."""

    def __init__(self, url, nome: str, historico_s: float = 0.0):
        super().__init__(daemon=True)
        self.url = int(url) if isinstance(url, str) and url.isdigit() else url
        self.nome, self.historico_s = nome, historico_s
        self._lock = threading.Lock()
        self._ult = None                        # (frame, ts)
        self._hist = collections.deque()        # (ts, frame)
        self.parar = threading.Event()
        self.conectou = threading.Event()
        self.terminou = False

    def run(self):
        eh_arquivo = isinstance(self.url, str) and os.path.isfile(self.url)
        avisou = False
        while not self.parar.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                if not avisou:
                    print(f"[{self.nome}] nao consegui abrir {self.url}; tentando de novo...")
                    avisou = True
                cap.release()
                time.sleep(2.0)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            print(f"[{self.nome}] conectado")
            avisou = False
            self.conectou.set()
            while not self.parar.is_set():
                t0 = time.time()
                ok, fr = cap.read()
                if not ok:
                    if eh_arquivo:
                        self.terminou = True
                        cap.release()
                        return
                    break                      # stream caiu: reconecta
                ts = time.time()
                with self._lock:
                    self._ult = (fr, ts)
                    if self.historico_s > 0:
                        self._hist.append((ts, fr))
                        while self._hist[0][0] < ts - self.historico_s:
                            self._hist.popleft()
                if eh_arquivo:                 # arquivo: simula tempo real
                    time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
            cap.release()
            if not self.parar.is_set():
                print(f"[{self.nome}] stream interrompido; reconectando...")
                time.sleep(1.0)

    def ultimo(self):
        with self._lock:
            return None if self._ult is None else (self._ult[0].copy(), self._ult[1])

    def mais_proximo(self, ts_alvo: float):
        with self._lock:
            if not self._hist:
                return None
            return min(self._hist, key=lambda p: abs(p[0] - ts_alvo))[1].copy()


def testar_streams(urls: list):
    """Tenta abrir cada URL, le um quadro, mostra a resolucao e salva probe_N.jpg.
    O termico parece mapa de calor e tem resolucao menor (C12: 384x288)."""
    urls = urls or [f"rtsp://192.168.144.108:554/stream={i}" for i in range(3)]  # stream=1 e o RGB
    ok_urls = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}] {url}")
        t0 = time.time()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        fr = None
        while cap.isOpened() and time.time() - t0 < 6.0:
            ok, f = cap.read()
            if ok:
                fr = f
                break
        cap.release()
        if fr is None:
            print(f"    FALHOU ({time.time() - t0:.1f}s)")
            continue
        cv2.imwrite(f"probe_{i}.jpg", fr)
        print(f"    OK: {fr.shape[1]}x{fr.shape[0]}, termico em cinza: {not modelos.termico_colorido(fr)} "
              f"-> probe_{i}.jpg")
        ok_urls.append(url)
    print("\nURLs que responderam:" if ok_urls else
          "\nNenhuma URL respondeu: camera ligada? PC na rede 192.168.144.x? Teste no VLC.")
    for u in ok_urls:
        print("  ", u)


# =============================================================================
# Instantaneo, gravacao bruta e amostrador
# =============================================================================
@dataclass
class Instantaneo:
    frame: np.ndarray
    ts: float                       # instante unix da captura do RGB
    termico: Optional[np.ndarray]
    gps: dict


class GravadorBruto:
    """Grava os instantaneos que entram na fila (rgb.mp4, termico.mp4, gps.csv; mesmo numero
    de quadros), para reprocessar depois com --reprocessar. O termico e repetido se faltar."""

    def __init__(self, pasta: str, fps: float):
        os.makedirs(pasta, exist_ok=True)
        self.pasta, self.fps = pasta, fps
        self.rgb = self.th = self.ultimo_th = None
        self.f = open(os.path.join(pasta, "gps.csv"), "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["quadro", "timestamp", "lat", "lon", "sat", "status"])
        self.n = 0

    def _writer(self, nome, img):
        h, w = img.shape[:2]
        return cv2.VideoWriter(os.path.join(self.pasta, nome), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def escrever(self, it: Instantaneo):
        if self.rgb is None:
            self.rgb = self._writer("rgb.mp4", it.frame)
        self.rgb.write(it.frame)
        if it.termico is not None and (self.ultimo_th is None or it.termico.shape == self.ultimo_th.shape):
            self.ultimo_th = it.termico
        if self.ultimo_th is not None:
            if self.th is None:
                self.th = self._writer("termico.mp4", self.ultimo_th)
            self.th.write(self.ultimo_th)
        g = it.gps
        self.w.writerow([self.n, f"{it.ts:.3f}", g["lat"], g["lon"], g["sat"], g["status"]])
        self.n += 1
        if self.n % 20 == 0:
            self.f.flush()

    def fechar(self):
        for x in (self.rgb, self.th):
            if x is not None:
                x.release()
        self.f.close()


class Amostrador(threading.Thread):
    """Tira instantaneos a cada 1/fps_proc s e poe na fila limitada (descarta o mais antigo)."""

    def __init__(self, rgb: FonteViva, termica: Optional[FonteViva], gps: GPSAoVivo,
                 fila: "queue.Queue", fps_proc: float, offset_s: float = 0.0,
                 gravador: Optional[GravadorBruto] = None):
        super().__init__(daemon=True)
        self.rgb, self.termica, self.gps, self.fila = rgb, termica, gps, fila
        self.fps_proc, self.offset_s, self.gravador = fps_proc, offset_s, gravador
        self.parar = threading.Event()
        self.amostrados = self.descartados = 0

    def _empilha(self, it):
        while True:
            try:
                self.fila.put_nowait(it)
                return
            except queue.Full:
                try:
                    self.fila.get_nowait()
                    self.descartados += 1
                except queue.Empty:
                    pass

    def run(self):
        periodo, prox, ultimo_ts = 1.0 / self.fps_proc, time.time(), None
        try:
            while not self.parar.is_set():
                agora = time.time()
                if agora < prox:
                    time.sleep(min(prox - agora, 0.02))
                    continue
                prox = max(prox + periodo, agora)      # sem rajada de recuperacao
                u = self.rgb.ultimo()
                if u is None or u[1] == ultimo_ts:     # nada novo
                    if self.rgb.terminou:
                        break
                    continue
                frame, ts = u
                ultimo_ts = ts
                th = None
                if self.termica is not None:
                    alvo = ts + self.offset_s           # offset > 0: o termico chega atrasado
                    if alvo > time.time():
                        time.sleep(min(alvo - time.time(), 1.0))
                    th = self.termica.mais_proximo(alvo)
                it = Instantaneo(frame, ts, th, self.gps.instantaneo(ts))
                if self.gravador is not None:
                    self.gravador.escrever(it)
                self._empilha(it)
                self.amostrados += 1
        finally:
            self._empilha(None)                        # sinal de fim para o processador


# =============================================================================
# Tracker por IoU
# =============================================================================
@dataclass
class Track:
    id: int
    cls_name: str
    box: tuple
    missed: int = 0
    run: dict = field(default_factory=dict)        # falha -> contador acumulado
    opened: set = field(default_factory=set)       # falhas que ja geraram evento


def iou(a: tuple, b: tuple) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class IoUTracker:
    """Associa deteccoes a tracks da mesma classe por sobreposicao (guloso, maior IoU primeiro).
    max_missed: quantos instantaneos um track sobrevive sem ser visto (evita evento duplicado)."""

    def __init__(self, iou_thr: float = 0.3, max_missed: int = 30):
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks: list = []
        self._next_id = 1

    def update(self, dets: list) -> list:
        pares = sorted(((iou(t.box, d.box), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(dets) if t.cls_name == d.cls_name), reverse=True)
        usados_t, usados_d, out = set(), set(), []
        for v, ti, di in pares:
            if v < self.iou_thr:
                break
            if ti in usados_t or di in usados_d:
                continue
            usados_t.add(ti)
            usados_d.add(di)
            t = self.tracks[ti]
            t.box, t.missed = dets[di].box, 0
            out.append((t, dets[di]))
        for ti, t in enumerate(self.tracks):
            if ti not in usados_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in usados_d:
                t = Track(self._next_id, d.cls_name, d.box)
                self._next_id += 1
                self.tracks.append(t)
                out.append((t, d))
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return out


# =============================================================================
# Registro de eventos
# =============================================================================
class EventLog:
    CAMPOS = ["evento_id", "track_id", "componente", "falha", "quadro", "t_inspecao_s",
              "timestamp_unix", "data_hora", "latitude", "longitude", "satelites", "gps_status",
              "score_anomalia", "contraste_hotspot_cinza", "imagem_rgb", "imagem_rgb_anotada",
              "imagem_termica", "imagem_termica_anotada"]

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.ev_dir = os.path.join(out_dir, "eventos")
        os.makedirs(self.ev_dir, exist_ok=True)
        self.eventos: list = []
        with open(self._csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CAMPOS).writeheader()

    @property
    def _csv(self):
        return os.path.join(self.out_dir, "eventos.csv")

    def _img(self, pasta, nome, img) -> str:
        """Salva e devolve o caminho RELATIVO a out_dir (o CSV continua valido se a pasta mudar)."""
        if img is None:
            return ""
        caminho = os.path.join(pasta, nome)
        return os.path.relpath(caminho, self.out_dir).replace(os.sep, "/") if modelos.imwrite_unicode(caminho, img) else ""

    def abrir(self, track: Track, falha: str, quadro: int, t: float, it: Instantaneo,
              score, delta_t, rgb_vis, th_vis) -> dict:
        n = len(self.eventos) + 1
        pasta = os.path.join(self.ev_dir, f"evt_{n:04d}_{track.cls_name}_{TAG_FALHA[falha]}")
        os.makedirs(pasta, exist_ok=True)
        g = it.gps
        ev = {"evento_id": n, "track_id": track.id, "componente": track.cls_name, "falha": falha,
              "quadro": quadro, "t_inspecao_s": round(t, 3), "timestamp_unix": round(it.ts, 3),
              "data_hora": datetime.fromtimestamp(it.ts).isoformat(timespec="seconds"),
              "latitude": g["lat"], "longitude": g["lon"], "satelites": g["sat"], "gps_status": g["status"],
              "score_anomalia": None if score is None else round(float(score), 4),
              "contraste_hotspot_cinza": None if delta_t is None else round(float(delta_t), 2),
              "imagem_rgb": self._img(pasta, "rgb.jpg", it.frame),
              "imagem_rgb_anotada": self._img(pasta, "rgb_anotado.jpg", rgb_vis),
              "imagem_termica": self._img(pasta, "termica.jpg", it.termico),
              "imagem_termica_anotada": self._img(pasta, "termica_anotada.jpg", th_vis)}
        with open(os.path.join(pasta, "evento.json"), "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        self.eventos.append(ev)
        # salva NA HORA: se o programa cair, os eventos anteriores ja estao no disco
        with open(self._csv, "a", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CAMPOS).writerow(ev)
        with open(os.path.join(self.out_dir, "eventos.json"), "w", encoding="utf-8") as f:
            json.dump(self.eventos, f, ensure_ascii=False, indent=2)
        return ev


# =============================================================================
# Desenho
# =============================================================================
def desenhar_rotulo(img, x, y, texto, cor):
    fonte, escala = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    (tw, th), base = cv2.getTextSize(texto, fonte, escala, 1)
    y0 = max(th + base + 2, y)
    cv2.rectangle(img, (x, y0 - th - base - 4), (x + tw + 6, y0), cor, -1)
    cv2.putText(img, texto, (x + 3, y0 - base - 2), fonte, escala, BRANCO, 1, cv2.LINE_AA)


def desenhar_caixa(img, det: Detection, ativas: list, track_id: int):
    x1, y1, x2, y2 = map(int, det.box)
    cor = VERMELHO if ativas else VERDE
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
    texto = f"{det.cls_name} #{track_id}" + ("".join(f" | {TAG_FALHA[f].upper()}" for f in ativas))
    desenhar_rotulo(img, x1, y1, texto, cor)


def desenhar_hud(img, t: float, gps: dict):
    g = {"ok": f"GPS {gps['lat']:.6f}, {gps['lon']:.6f}" if gps["status"] == "ok" else "",
         "sem_fix": "GPS sem fix", "sem_log": "GPS: sem log"}[gps["status"]]
    texto, y = f"t={t:6.1f}s  {g}", img.shape[0] - 12
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRETO, 3, cv2.LINE_AA)
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1, cv2.LINE_AA)


def desenhar_termico(th_vis, termo: dict, ativas: list):
    cor = VERMELHO if ativas else VERDE
    cv2.polylines(th_vis, [np.array(termo["quad"], np.int32)], True, cor, 1)
    x1, y1, x2, y2 = map(int, termo["caixa"])
    cv2.rectangle(th_vis, (x1, y1), (x2, y2), cor, 1)
    if termo["roi_mask"] is not None:
        c, _ = cv2.findContours(termo["roi_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(th_vis, c, -1, VERDE, 1)
    for h in termo["hotspots"]:
        c, _ = cv2.findContours(h.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(th_vis, c, -1, VERMELHO, 1)


# =============================================================================
# Processador: um instantaneo por vez
# =============================================================================
class Processador:
    """Recebe instantaneos (ao vivo ou de uma gravacao bruta) e gera os eventos.
    scorers: {classe: score(frame, box)}; limiares: {classe: limiar} (senao score_threshold).
    O ITADA roda em TODAS as classes detectadas, se houver termico e homografia."""

    def __init__(self, detect: Callable, scorers: dict, out_dir: str, H: Optional[np.ndarray] = None,
                 score_threshold: float = 0.5, limiares: Optional[dict] = None, min_frames: int = 3,
                 track_memory: int = 30, roi_margin: float = 0.15, roi_refino: bool = False,
                 filtro_fundo: Optional[float] = None, fps_video: float = 3.0):
        os.makedirs(out_dir, exist_ok=True)
        self.detect, self.scorers, self.H = detect, scorers or {}, H
        self.score_threshold, self.limiares = score_threshold, limiares or {}
        self.min_frames, self.roi_margin, self.roi_refino = min_frames, roi_margin, roi_refino
        self.filtro_fundo = filtro_fundo
        self.tracker = IoUTracker(max_missed=track_memory)
        self.log = EventLog(out_dir)
        self.out_dir, self.fps_video = out_dir, fps_video
        self.writer = None
        self.n = 0
        self.t0 = None
        self._avisou_cor = False

    def processar(self, it: Instantaneo):
        if self.t0 is None:
            self.t0 = it.ts
        t, quadro = it.ts - self.t0, self.n
        self.n += 1
        vis = it.frame.copy()
        th_vis = it.termico.copy() if it.termico is not None else None
        usar_termico = it.termico is not None and self.H is not None
        if usar_termico and not self._avisou_cor:
            self._avisou_cor = True
            if modelos.termico_colorido(it.termico):
                print("[AVISO] o termico tem PALETA DE COR: configure a C12 em WHITE-HOT, "
                      "senao o ITADA nao e confiavel.")

        novos = []
        for track, det in self.tracker.update(self.detect(it.frame)):
            agora = {}
            score = None
            if det.cls_name in self.scorers:
                score = self.scorers[det.cls_name](it.frame, det.box)
                if score >= self.limiares.get(det.cls_name, self.score_threshold):
                    agora[FALHA_ANOMALIA] = score
            termo = None
            if usar_termico:
                termo = modelos.avaliar_hotspot(it.termico, det.box, self.H, self.roi_margin, self.roi_refino,
                                                self.filtro_fundo)
                if termo["hotspot"]:
                    agora[FALHA_HOTSPOT] = termo["delta_t"]

            # confirmacao temporal: sobe 1 com falha, desce 1 sem; teto para a caixa
            # voltar a ficar verde poucos instantaneos depois que a falha some
            for falha in (FALHA_ANOMALIA, FALHA_HOTSPOT):
                atual = track.run.get(falha, 0)
                track.run[falha] = min(atual + 1, self.min_frames + 2) if falha in agora else max(0, atual - 1)
            ativas = [f for f in (FALHA_ANOMALIA, FALHA_HOTSPOT) if track.run[f] >= self.min_frames]

            desenhar_caixa(vis, det, ativas, track.id)
            if termo is not None:
                desenhar_termico(th_vis, termo, ativas)
                for cx, cy in termo["centroides_rgb"]:
                    cv2.circle(vis, (int(cx), int(cy)), 5, VERMELHO, 2)
            for falha in ativas:
                if falha not in track.opened:
                    track.opened.add(falha)
                    novos.append((track, falha, score, termo["delta_t"] if termo else None))

        desenhar_hud(vis, t, it.gps)
        for track, falha, score, dt in novos:
            ev = self.log.abrir(track, falha, quadro, t, it, score, dt, vis, th_vis)
            print(f"[EVENTO {ev['evento_id']}] t={t:.1f}s {track.cls_name} #{track.id}: {falha} "
                  f"| GPS {it.gps['status']}")
        if self.writer is None:
            h, w = vis.shape[:2]
            self.writer = cv2.VideoWriter(os.path.join(self.out_dir, "video_anotado.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), self.fps_video, (w, h))
        self.writer.write(vis)

    def fechar(self):
        if self.writer is not None:
            self.writer.release()


# =============================================================================
# Execucao ao vivo
# =============================================================================
def executar_inspecao(detect: Callable, scorers: dict, out_dir: str, rgb_url, thermal_url=None,
                      gps_linhas: Optional[Iterable[str]] = None, H: Optional[np.ndarray] = None,
                      fps_proc: float = 3.0, fila_max: int = 8, thermal_offset: float = 0.0,
                      gravar_bruto: bool = False, duracao_s: Optional[float] = None,
                      espera_conexao_s: float = 20.0, gps_max_idade_s: float = 5.0, **proc_kw) -> dict:
    """Roda ate Ctrl+C, fim do arquivo (modo teste) ou duracao_s. proc_kw vai para o Processador.
    -> {'eventos', 'amostrados', 'descartados', 'out_dir', 'log'}"""
    os.makedirs(out_dir, exist_ok=True)
    rgb = FonteViva(rgb_url, "RGB")
    termica = FonteViva(thermal_url, "TERMICO", 3.0 + abs(thermal_offset)) if thermal_url is not None else None
    gps = GPSAoVivo(gps_linhas, gps_max_idade_s)
    for th in (rgb, termica, gps):
        if th is not None:
            th.start()

    t0 = time.time()
    while rgb.ultimo() is None:
        if time.time() - t0 > espera_conexao_s:
            rgb.parar.set()
            raise RuntimeError(f"O stream RGB nao entregou imagem em {espera_conexao_s:.0f}s: {rgb_url}")
        time.sleep(0.1)
    if termica is not None and not termica.conectou.wait(espera_conexao_s):
        print("[AVISO] stream termico nao conectou; seguindo so com o RGB (sem ITADA) ate ele chegar.")

    fila: "queue.Queue" = queue.Queue(maxsize=fila_max)
    grav = GravadorBruto(os.path.join(out_dir, "bruto"), fps_proc) if gravar_bruto else None
    amostrador = Amostrador(rgb, termica, gps, fila, fps_proc, thermal_offset, grav)
    proc = Processador(detect, scorers, out_dir, H, fps_video=fps_proc, **proc_kw)
    parar = threading.Event()
    if duracao_s:
        tmr = threading.Timer(duracao_s, parar.set)
        tmr.daemon = True
        tmr.start()

    amostrador.start()
    print(f"[BASE] inspecao iniciada: {fps_proc:g} quadros/s, fila de {fila_max}. Ctrl+C encerra. Saida: {out_dir}")
    try:
        while not parar.is_set():
            try:
                it = fila.get(timeout=0.2)
            except queue.Empty:
                continue
            if it is None:
                break
            proc.processar(it)
    except KeyboardInterrupt:
        print("\n[BASE] interrompido pelo usuario; eventos ja salvos.")
    finally:
        amostrador.parar.set()
        rgb.parar.set()
        if termica is not None:
            termica.parar.set()
        amostrador.join(timeout=2.0)
        if grav is not None:
            grav.fechar()
        proc.fechar()
    n_ev = len(proc.log.eventos)
    print(f"[BASE] fim: {amostrador.amostrados} quadros amostrados, {amostrador.descartados} descartados "
          f"por fila cheia, {n_ev} evento(s).")
    return {"eventos": n_ev, "amostrados": amostrador.amostrados, "descartados": amostrador.descartados,
            "out_dir": out_dir, "log": proc.log}


# =============================================================================
# Reprocessar uma gravacao bruta (sem tempo real, quadro a quadro)
# =============================================================================
def reprocessar(pasta_bruto: str, detect: Callable, scorers: dict, out_dir: str,
                H: Optional[np.ndarray] = None, **proc_kw):
    """Le bruto/rgb.mp4, termico.mp4 e gps.csv (mesmo indice de quadro) e passa cada quadro
    pelo Processador, como na inspecao ao vivo. -> EventLog"""
    with open(os.path.join(pasta_bruto, "gps.csv"), encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    num = lambda v, tipo=float: tipo(float(v)) if v not in (None, "", "None") else None
    cap = cv2.VideoCapture(os.path.join(pasta_bruto, "rgb.mp4"))
    p_th = os.path.join(pasta_bruto, "termico.mp4")
    cap_th = cv2.VideoCapture(p_th) if os.path.exists(p_th) else None
    if not cap.isOpened():
        raise FileNotFoundError(os.path.join(pasta_bruto, "rgb.mp4"))
    proc = Processador(detect, scorers, out_dir, H, fps_video=cap.get(cv2.CAP_PROP_FPS) or 3.0, **proc_kw)
    try:
        for row in linhas:
            ok, fr = cap.read()
            if not ok:
                break
            th = None
            if cap_th is not None:
                ok_t, th = cap_th.read()
                th = th if ok_t else None
            gps = {"lat": num(row["lat"]), "lon": num(row["lon"]), "sat": num(row["sat"], int),
                   "status": row["status"]}
            proc.processar(Instantaneo(fr, float(row["timestamp"]), th, gps))
    finally:
        cap.release()
        if cap_th is not None:
            cap_th.release()
        proc.fechar()
    print(f"[REPROCESSAR] {proc.n} quadros, {len(proc.log.eventos)} evento(s). Saida: {out_dir}")
    return proc.log


# =============================================================================
# Linha de comando
# =============================================================================
def _pares(lista: list, nome: str, tipo=str) -> dict:
    """['classe=valor', ...] -> {classe: valor}"""
    out = {}
    for item in lista or []:
        if "=" not in item:
            raise SystemExit(f"{nome}: use CLASSE=VALOR (veio '{item}')")
        k, v = item.split("=", 1)
        out[k.strip()] = tipo(v.strip())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testar-streams", nargs="*", default=None, metavar="URL",
                   help="so testa as URLs dos streams e sai (sem URL: testa stream=0/1/2 da C12)")
    p.add_argument("--reprocessar", default=None, metavar="PASTA_BRUTO",
                   help="reprocessa uma gravacao bruta (rgb.mp4, termico.mp4, gps.csv) em vez de ler ao vivo")
    p.add_argument("--yolo-weights")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--classes", default=None, help="classes a analisar, separadas por virgula (padrao: todas)")
    p.add_argument("--patchcore", action="append", metavar="CLASSE=CKPT",
                   help="checkpoint do PatchCore de uma classe (repita para cada classe)")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="limiar padrao do PatchCore (calibre com avaliar.py anomalia)")
    p.add_argument("--limiar", action="append", metavar="CLASSE=VALOR", help="limiar de uma classe especifica")
    p.add_argument("--recorte-margem", type=float, default=0.0,
                   help="margem do recorte quadrado do PatchCore (0 = lado maior da caixa)")
    p.add_argument("--rgb-url", default="rtsp://192.168.144.108:554/stream=1")
    p.add_argument("--thermal-url", default=None, help="stream termico (descubra com --testar-streams)")
    p.add_argument("--homography", default=None, help="homografia_c12.npy (modelos.py calibrar)")
    p.add_argument("--thermal-offset", type=float, default=0.0,
                   help="latencia do termico em s (positivo = chega ATRASADO em relacao ao RGB)")
    p.add_argument("--gps-port", default=None, help="porta serial do LoRa (ex.: COM5). Sem ela: gps 'sem_log'")
    p.add_argument("--gps-baud", type=int, default=115200)
    p.add_argument("--gps-max-idade", type=float, default=5.0, help="posicao mais velha que isso (s) = sem_fix")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fps-proc", type=float, default=3.0, help="quadros processados por segundo")
    p.add_argument("--fila-max", type=int, default=8)
    p.add_argument("--gravar-bruto", action="store_true", help="grava os quadros processados para --reprocessar")
    p.add_argument("--duracao", type=float, default=None, help="encerra apos N segundos")
    p.add_argument("--min-frames", type=int, default=3, help="instantaneos com falha para confirmar o evento")
    p.add_argument("--track-memory", type=int, default=30,
                   help="instantaneos que um componente pode sumir e continuar sendo o MESMO")
    p.add_argument("--roi-margin", type=float, default=0.15, help="expansao (fracao) da caixa projetada no termico")
    p.add_argument("--roi-refino", action="store_true",
                   help="delimita o componente dentro da caixa projetada por Otsu (Zaripova et al. 2019)")
    p.add_argument("--filtro-fundo", type=float, default=None, metavar="MARGEM",
                   help="descarta hotspot cujo pico nao passa da mediana do fundo ao redor da ROI + MARGEM "
                        "(niveis de cinza). Elimina a borda borrada de componente mais frio que o fundo")
    a = p.parse_args()

    if a.testar_streams is not None:
        testar_streams(a.testar_streams)
        return
    if not a.yolo_weights:
        p.error("--yolo-weights e obrigatorio")

    classes = {c.strip() for c in a.classes.split(",")} if a.classes else None
    detect = modelos.carregar_detector(a.yolo_weights, a.conf, classes)
    scorers = modelos.carregar_patchcores(_pares(a.patchcore, "--patchcore"), a.recorte_margem)
    H = np.load(a.homography) if a.homography else None
    proc_kw = dict(score_threshold=a.score_threshold, limiares=_pares(a.limiar, "--limiar", float),
                   min_frames=a.min_frames, track_memory=a.track_memory,
                   roi_margin=a.roi_margin, roi_refino=a.roi_refino, filtro_fundo=a.filtro_fundo)
    print(f"[BASE] PatchCore em: {', '.join(scorers) or 'nenhuma classe'}")

    if a.reprocessar:
        reprocessar(a.reprocessar, detect, scorers, a.out_dir or "reprocessado", H, **proc_kw)
        return
    if a.thermal_url and H is None:
        print("[AVISO] --thermal-url sem --homography: o ITADA sera pulado.")
    out_dir = a.out_dir or datetime.now().strftime("inspecao_%Y%m%d_%H%M%S")
    gps_linhas = linhas_serial(a.gps_port, a.gps_baud) if a.gps_port else None
    executar_inspecao(detect, scorers, out_dir, a.rgb_url, a.thermal_url, gps_linhas, H,
                      fps_proc=a.fps_proc, fila_max=a.fila_max, thermal_offset=a.thermal_offset,
                      gravar_bruto=a.gravar_bruto, duracao_s=a.duracao, gps_max_idade_s=a.gps_max_idade,
                      **proc_kw)


if __name__ == "__main__":
    main()
