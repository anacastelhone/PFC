"""
Teste da base de inspecao (base_inspecao.py) SEM robo, SEM serial e SEM modelos:
  - o "stream" RGB e o termico sao arquivos de video sinteticos, reproduzidos em tempo real;
  - o GPS e um gerador de linhas no formato do LoRa ("LAT:..,LON:..,SAT:..");
  - detector e PatchCore sao stand-ins (nao testam a YOLO nem o PatchCore reais).

Verifica: eventos de anomalia e de hotspot, 1 evento por falha, GPS no evento, eventos salvos
AO LONGO da execucao (antes do fim), fila que descarta quando o processamento atrasa, GPS ausente/velho,
gravacao bruta reprocessavel (--reprocessar), PatchCore so nas classes com checkpoint e parada por --duracao.

USO: python test_base_inspecao.py [pasta_de_saida]
"""
import os
import sys
import csv
import json
import time
import shutil
import threading
import numpy as np
import cv2

import base_inspecao as bi
from base_inspecao import FALHA_ANOMALIA, FALHA_HOTSPOT
from modelos import Detection

W, H_RGB = 640, 360
TW, TH = 384, 288
N = 80
FPS = 20
HOM = np.array([[0.6, 0, 20], [0, 0.6, 10], [0, 0, 1]], dtype=np.float64)
ANOM_FRAMES = range(20, 46)
HOT_FRAMES = range(25, 61)


def box_no_frame(i):
    x = 300 + int(0.5 * i)
    return (x, 100, x + 40, 250)


def gerar_videos(pasta):
    rng = np.random.default_rng(0)
    rgb_path, th_path = os.path.join(pasta, "rgb.mp4"), os.path.join(pasta, "termico.mp4")
    wr = cv2.VideoWriter(rgb_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_RGB))
    wt = cv2.VideoWriter(th_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (TW, TH))
    for i in range(N):
        fr = np.clip(rng.normal(90, 6, (H_RGB, W, 3)), 0, 255).astype(np.uint8)
        x1, y1, x2, y2 = box_no_frame(i)
        fr[y1:y2, x1:x2] = 200
        if i in ANOM_FRAMES:
            cv2.line(fr, (x1 + 5, y1 + 60), (x2 - 5, y1 + 70), (10, 10, 10), 3)
        wr.write(fr)
        th = np.clip(rng.normal(80, 3, (TH, TW)), 0, 255).astype(np.uint8)
        if i in HOT_FRAMES:
            cx, cy = int(0.6 * (x1 + x2) / 2 + 20), int(0.6 * (y1 + 20) + 10)
            th[cy - 4:cy + 4, cx - 4:cx + 4] = 200
        wt.write(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    wr.release()
    wt.release()
    return rgb_path, th_path


def detector_por_conteudo(atraso_s=0.0, classe="glass-insulator"):
    """Acha a 'caixa' clara (200) no quadro, qualquer que seja o quadro amostrado."""
    def detect(frame):
        if atraso_s:
            time.sleep(atraso_s)
        g = frame[..., 1]
        ys, xs = np.where(g > 150)
        if xs.size < 200:
            return []
        return [Detection(classe, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), 0.9)]
    return detect


def scorer_falso(frame, box):
    x1, y1, x2, y2 = map(int, box)
    return 0.9 if frame[y1:y2, x1:x2].min() < 40 else 0.1


SCORERS = {"glass-insulator": scorer_falso}


def gps_linhas(com_fix=True, so_zero=False, parar_apos_s=None):
    """Gerador de linhas do LoRa. so_zero: manda 0,0 (sem fix). parar_apos_s: depois disso nao manda mais."""
    t0 = time.time()
    i = 0
    while True:
        if parar_apos_s is not None and time.time() - t0 > parar_apos_s:
            time.sleep(0.1)
            continue
        if so_zero:
            yield "LAT:0.0,LON:0.0,SAT:0"
        else:
            yield f"LAT:{-19.74700 - 0.00001 * i:.5f},LON:{-47.93800:.5f},SAT:8"
        i += 1
        time.sleep(0.1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "teste_base_saida"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    rgb, th = gerar_videos(out)
    falhas = []

    def check(cond, msg):
        print(("OK   - " if cond else "FALHA - ") + msg)
        if not cond:
            falhas.append(msg)

    # ---- parse do GPS ----
    check(bi.parse_gps("LAT:-19.74723,LON:-47.93811,SAT:8") == (-19.74723, -47.93811, 8), "parse_gps le o formato do LoRa")
    check(bi.parse_gps("lixo") is None and bi.parse_gps("") is None, "parse_gps ignora linha invalida")

    # ---- 1) corrida completa, com leitura do disco DURANTE a execucao ----
    res1 = os.path.join(out, "r1")
    visto_durante = {}

    def espiao():
        for _ in range(400):
            f = os.path.join(res1, "eventos.csv")
            if os.path.exists(f):
                with open(f, encoding="utf-8-sig") as fh:
                    n = len(list(csv.DictReader(fh)))
                if n >= 1:
                    visto_durante["n"] = n
                    visto_durante["t"] = time.time()
                    return
            time.sleep(0.05)

    t_ini = time.time()
    threading.Thread(target=espiao, daemon=True).start()
    r = bi.executar_inspecao(detector_por_conteudo(), SCORERS, res1, rgb, th, gps_linhas(), HOM,
                             fps_proc=10, fila_max=8, min_frames=3, gravar_bruto=True)
    t_fim = time.time()
    ev = {e["falha"]: e for e in r["log"].eventos}
    check(r["eventos"] == 2 and FALHA_ANOMALIA in ev and FALHA_HOTSPOT in ev,
          f"1 evento de anomalia + 1 de hotspot; veio {r['eventos']}")
    check("n" in visto_durante and visto_durante["t"] < t_fim - 0.3,
          "eventos.csv ja tinha evento ANTES do fim da execucao (salvo ao longo da inspecao)")
    check(r["amostrados"] >= 30, f"amostrou os quadros em tempo real (~10/s): {r['amostrados']}")
    check(all(e["gps_status"] == "ok" and e["latitude"] is not None and e["latitude"] < -19.74 for e in r["log"].eventos),
          "GPS ao vivo gravado nos eventos (status ok)")
    check(all(e.get("timestamp_unix") and abs(e["timestamp_unix"] - t_ini) < 30 for e in r["log"].eventos),
          "timestamp_unix presente e plausivel")
    for e in r["log"].eventos:
        pasta = os.path.join(res1, "eventos", f"evt_{e['evento_id']:04d}_glass-insulator_{bi.TAG_FALHA[e['falha']]}")
        check(all(os.path.exists(os.path.join(pasta, a)) for a in ("rgb.jpg", "rgb_anotado.jpg", "termica.jpg", "evento.json")),
              f"evento {e['evento_id']} salvou rgb, termica e evento.json")
    check(os.path.exists(os.path.join(res1, "video_anotado.mp4")), "video_anotado.mp4 gerado")

    # ---- 2) gravacao bruta reprocessavel offline ----
    bruto = os.path.join(res1, "bruto")
    check(all(os.path.exists(os.path.join(bruto, a)) for a in ("rgb.mp4", "termico.mp4", "gps.csv")),
          "gravacao bruta: rgb.mp4, termico.mp4, gps.csv")
    off = os.path.join(out, "offline")
    logo = bi.reprocessar(bruto, detector_por_conteudo(), SCORERS, off, HOM, min_frames=3)
    check({e["falha"] for e in logo.eventos} == {FALHA_ANOMALIA, FALHA_HOTSPOT} and len(logo.eventos) == 2,
          "reprocessar o bruto (--reprocessar) da os mesmos 2 eventos")
    check(all(e["gps_status"] == "ok" for e in logo.eventos), "o gps.csv bruto casa com o video no reprocessamento")

    # ---- 3) processamento lento: a fila descarta e a inspecao termina ----
    res3 = os.path.join(out, "r3")
    t0 = time.time()
    r3 = bi.executar_inspecao(detector_por_conteudo(atraso_s=0.4), SCORERS, res3, rgb, th, gps_linhas(), HOM,
                              fps_proc=10, fila_max=2, min_frames=3)
    check(r3["descartados"] > 5, f"fila cheia descarta os instantaneos mais antigos: {r3['descartados']} descartes")
    check(time.time() - t0 < 25, "inspecao com processamento lento termina (nao trava)")

    # ---- 4) GPS ausente e GPS velho ----
    r4 = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r4"), rgb, th, None, HOM, fps_proc=10)
    check(r4["eventos"] == 2 and all(e["gps_status"] == "sem_log" for e in r4["log"].eventos),
          "sem GPS configurado: eventos salvos com gps_status=sem_log")
    r5 = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r5"), rgb, th,
                              gps_linhas(so_zero=True), HOM, fps_proc=10)
    check(r5["eventos"] == 2 and all(e["gps_status"] == "sem_fix" for e in r5["log"].eventos),
          "GPS so com 0,0 (sem fix): eventos com gps_status=sem_fix")
    r6 = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r6"), rgb, th,
                              gps_linhas(parar_apos_s=0.3), HOM, fps_proc=10, gps_max_idade_s=1.0)
    check(r6["eventos"] == 2 and all(e["gps_status"] == "sem_fix" for e in r6["log"].eventos),
          "GPS que parou de chegar (mais velho que o limite): gps_status=sem_fix nos eventos tardios")

    # ---- 5) sem termico: so a anomalia ----
    r7 = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r7"), rgb, None, gps_linhas(), None, fps_proc=10)
    check(r7["eventos"] == 1 and r7["log"].eventos[0]["falha"] == FALHA_ANOMALIA, "sem stream termico: so o evento de anomalia")

    # ---- 5b) classe sem PatchCore (ex.: espacador): so o hotspot ----
    r7b = bi.executar_inspecao(detector_por_conteudo(classe="spacer"), SCORERS, os.path.join(out, "r7b"), rgb, th,
                               gps_linhas(), HOM, fps_proc=10)
    check(r7b["eventos"] == 1 and r7b["log"].eventos[0]["falha"] == FALHA_HOTSPOT,
          "classe sem checkpoint do PatchCore: so o evento de hotspot")

    # ---- 5c) limiar por classe ----
    r7c = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r7c"), rgb, th, gps_linhas(), HOM,
                               fps_proc=10, limiares={"glass-insulator": 0.95})
    check(r7c["eventos"] == 1 and r7c["log"].eventos[0]["falha"] == FALHA_HOTSPOT,
          "limiar por classe acima do score: anomalia nao dispara")

    # ---- 6) --duracao encerra antes do fim do video ----
    t0 = time.time()
    r8 = bi.executar_inspecao(detector_por_conteudo(), SCORERS, os.path.join(out, "r8"), rgb, th, gps_linhas(), HOM,
                              fps_proc=10, duracao_s=1.0)
    check(time.time() - t0 < 3.5 and r8["amostrados"] < 20, f"duracao=1s encerra cedo ({time.time() - t0:.1f}s, {r8['amostrados']} quadros)")

    # ---- 7) stream que nao existe: erro claro ----
    try:
        bi.executar_inspecao(detector_por_conteudo(), {}, os.path.join(out, "r9"), "arquivo_que_nao_existe.mp4", None,
                             None, None, espera_conexao_s=3.0)
        check(False, "stream inexistente deve dar erro claro")
    except RuntimeError as e:
        check("nao entregou imagem" in str(e), "stream inexistente: erro claro depois da espera")

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S):")
        for f in falhas:
            print("  -", f)
        sys.exit(1)
    print("TUDO OK")


if __name__ == "__main__":
    main()
