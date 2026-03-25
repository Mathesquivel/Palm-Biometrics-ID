"""
PalmBiometrics — Reconhecimento de veias da palma da mão
=========================================================
Fluxo de cadastro:
  1. Captura N imagens da palma
  2. Aplica pré-processamento (realce de veias)
  3. Extrai template biométrico LBP (irreconstruível)
  4. Salva APENAS os templates em JSON — nenhuma imagem fica em disco

Fluxo de reconhecimento:
  1. Captura uma imagem
  2. Extrai template
  3. Compara por similaridade de cosseno com todos os usuários cadastrados
  4. Aceita se o melhor score ≥ threshold configurado em biometric_template.py
"""

import os
import cv2
import shutil
import numpy as np

from leds import setup, liga_leds, desliga_leds, cleanup
from camera import setup_camera, start_preview, stop_preview
from preprocess_veins import preprocess_veins
from biometric_template import (
    extract_template, match_template,
    save_templates, load_templates,
    MATCH_THRESHOLD,
)
from interface import wait_for_hand

USUARIOS_DIR = "USUARIOS"
NUM_CAPTURAS = 16  # amostras por cadastro (mais = maior robustez)


def _capture_guided(picam2, msg_prefix: str = "") -> np.ndarray | None:
    """Captura com interface de guia. Retorna imagem cinza ou None se cancelado."""
    if msg_prefix:
        print(msg_prefix)
    return wait_for_hand(picam2)


def _templates_path(nome: str) -> str:
    return os.path.join(USUARIOS_DIR, nome, "templates.json")


def cadastrar(picam2) -> None:
    from database import add_user, list_users

    nome = input("Nome para cadastro: ").strip()
    if not nome:
        print("Nome inválido.")
        return

    usuarios_existentes = [n for _, n in list_users()]
    if nome in usuarios_existentes:
        print(f"Usuário '{nome}' já existe.")
        return

    user_dir = os.path.join(USUARIOS_DIR, nome)
    os.makedirs(user_dir, exist_ok=True)

    templates = []
    print(f"\nCadastrando {NUM_CAPTURAS} amostras.")
    print("Siga as instruções na tela. Pressione ESC para cancelar.\n")

    for i in range(NUM_CAPTURAS):
        print(f"  [{i + 1}/{NUM_CAPTURAS}] Posicione a mão...")
        gray = _capture_guided(picam2)
        if gray is None:
            print("Cadastro cancelado.")
            return

        processed = preprocess_veins(gray)
        template = extract_template(processed)
        templates.append(template)
        print(f"  [{i + 1}/{NUM_CAPTURAS}] Template extraído. (nenhuma imagem salva)")

    save_templates(templates, _templates_path(nome))
    add_user(nome)
    print(f"\nUsuário '{nome}' cadastrado com {NUM_CAPTURAS} templates biométricos.")


def reconhecer(picam2) -> None:
    if not os.path.isdir(USUARIOS_DIR):
        print("Nenhum usuário cadastrado.")
        return

    usuarios = [
        u for u in os.listdir(USUARIOS_DIR)
        if os.path.exists(_templates_path(u))
    ]
    if not usuarios:
        print("Nenhum usuário cadastrado.")
        return

    gray = _capture_guided(picam2, "Posicione a palma na área indicada. ESC para cancelar.")
    if gray is None:
        print("Reconhecimento cancelado.")
        return
    processed = preprocess_veins(gray)
    probe = extract_template(processed)

    melhor_usuario = None
    melhor_score = 0.0
    melhor_match = False

    for usuario in usuarios:
        stored = load_templates(_templates_path(usuario))
        matched, score = match_template(probe, stored)
        if score > melhor_score:
            melhor_score = score
            melhor_usuario = usuario
            melhor_match = matched

    if melhor_match:
        print(f"\nBem-vindo, {melhor_usuario}!  (score: {melhor_score:.3f})")
    else:
        score_str = f"{melhor_score:.3f}" if melhor_usuario else "N/A"
        print(f"\nAcesso negado — mão não reconhecida.  (melhor score: {score_str}, threshold: {MATCH_THRESHOLD})")


def listar_usuarios() -> None:
    from database import list_users
    users = list_users()
    if users:
        print("\nUsuários cadastrados:")
        for _, nome in users:
            tmpl = _templates_path(nome)
            status = f"{len(load_templates(tmpl))} templates" if os.path.exists(tmpl) else "sem templates"
            print(f"  - {nome}  ({status})")
    else:
        print("Nenhum usuário cadastrado.")


def remover_usuario() -> None:
    from database import remove_user, list_users
    users = list_users()
    if not users:
        print("Nenhum usuário cadastrado.")
        return

    listar_usuarios()
    nome = input("Nome do usuário a remover: ").strip()
    if not any(n == nome for _, n in users):
        print(f"Usuário '{nome}' não encontrado.")
        return

    confirmacao = input(f"Remover '{nome}' e todos os seus templates? [s/N] ").strip().lower()
    if confirmacao != "s":
        print("Cancelado.")
        return

    user_dir = os.path.join(USUARIOS_DIR, nome)
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir)
    remove_user(nome)
    print(f"Usuário '{nome}' removido.")


def menu() -> None:
    from database import init_db
    init_db()
    setup()
    picam2 = setup_camera()

    try:
        liga_leds()
        start_preview(picam2)

        while True:
            print("\n--- PalmBiometrics ---")
            print("1 - Cadastrar usuário")
            print("2 - Reconhecer mão")
            print("3 - Listar usuários")
            print("4 - Remover usuário")
            print("0 - Sair")
            opcao = input("Escolha: ").strip()

            if opcao == "1":
                cadastrar(picam2)
            elif opcao == "2":
                reconhecer(picam2)
            elif opcao == "3":
                listar_usuarios()
            elif opcao == "4":
                remover_usuario()
            elif opcao == "0":
                break
            else:
                print("Opção inválida.")

        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        desliga_leds()
        stop_preview(picam2)
        cleanup()


if __name__ == "__main__":
    menu()
