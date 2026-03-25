# Palm Biometrics ID

Sistema de identificação biométrica por **veias da palma da mão**, desenvolvido para rodar em arquitetura distribuída:

- **Raspberry Pi Zero** → captura de imagem + controle de LEDs IR (servidor leve)
- **PC / Notebook** → todo o processamento biométrico + interface gráfica (cliente)

---

## Arquitetura

```
┌─────────────────────────┐        WiFi / LAN        ┌──────────────────────────────┐
│   Raspberry Pi Zero     │ ◄────────────────────────► │        PC / Notebook         │
│                         │                            │                              │
│  • Picamera2 (IR)       │   HTTP (Flask :5000)       │  • Interface gráfica (PyQt5) │
│  • LEDs IR (GPIO 17)    │   GET  /frame              │  • Pré-processamento OpenCV  │
│  • server.py            │   POST /leds/on            │  • Template LBP biométrico   │
│                         │   POST /leds/off           │  • Matching por cosseno      │
│                         │   GET  /status             │  • Banco SQLite de usuários  │
└─────────────────────────┘                            └──────────────────────────────┘
```

---

## Hardware necessário

| Componente | Observação |
|---|---|
| Raspberry Pi Zero W / 2W | Recomendado: 2W (mais rápido) |
| Câmera NoIR para Raspberry Pi | Qualquer versão compatível |
| LEDs IR 850nm | 4–8 LEDs, resistores adequados |
| Transistor NPN (ex: BC337) | Para controlar os LEDs via GPIO 17 |
| PC com Windows / Linux / macOS | Para rodar a interface principal |
| Rede WiFi local (mesma rede) | Pi Zero e PC precisam estar na mesma rede |

---

## Instalação

### 1. Raspberry Pi Zero

```bash
# Clone o repositório no Pi Zero
git clone https://github.com/Mathesquivel/Palm-Biometrics-ID.git
cd "Palm-Biometrics-ID/pi_zero"

# Execute o instalador (requer sudo)
chmod +x install.sh
sudo ./install.sh
```

O instalador vai:
- Instalar todas as dependências
- Configurar o servidor para iniciar automaticamente no boot
- Exibir o IP do Pi Zero na rede

> **Anote o IP exibido** — você vai precisar dele no próximo passo.

Verificar se está rodando:
```bash
systemctl status palm-pizero
# ou
curl http://localhost:5000/status
```

---

### 2. PC (interface principal)

```bash
# Clone o repositório
git clone https://github.com/Mathesquivel/Palm-Biometrics-ID.git
cd "Palm-Biometrics-ID"

# Crie e ative ambiente virtual
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

**Configure o IP do Pi Zero em `config.py`:**

```python
PI_ZERO_HOST = "192.168.1.XXX"   # << coloque o IP do seu Pi Zero aqui
```

---

## Como rodar

### Interface gráfica (recomendado)

```bash
# Linux / macOS
./launch.sh

# Windows ou manual
python app.py
```

### Interface por linha de comando

```bash
python main.py
```

---

## Funcionalidades

### Cadastro de usuário
1. Informe o nome do usuário
2. Posicione a palma na elipse indicada na tela
3. O sistema captura automaticamente 16 amostras
4. Extrai templates biométricos LBP (nenhuma imagem é salva em disco)
5. Armazena apenas os templates no banco de dados

### Reconhecimento
1. Posicione a palma na elipse
2. O sistema captura e extrai o template
3. Compara com todos os usuários cadastrados via similaridade de cosseno
4. Aceita ou nega o acesso com base no threshold configurado

### Interface de guia (overlay)
- Elipse alvo centralizada
- Setas indicando direção de ajuste
- Indicadores de distância, foco e centralização
- Barra de progresso: captura automática após mão estável por 1,5 s
- Painel de status em tempo real

---

## Treinar o detector de mão (HOG + SVM)

O detector mão/não-mão melhora a precisão rejeitando objetos que não são mãos.
É opcional — o sistema funciona sem ele (usa fallback por contorno).

```bash
# 1. Capturar dataset de treino (precisa do Pi Zero conectado)
python capturar_treino_detector.py

# 2. Treinar o modelo SVM
python treinar_detector.py
# Gera: detector_mao.xml
```

---

## Estrutura do projeto

```
Palm-Biometrics-ID/
├── app.py                    # Interface gráfica principal (PyQt5)
├── main.py                   # Interface por linha de comando
├── config.py                 # Configurações (IP do Pi Zero, timeouts)
├── camera.py                 # Cliente HTTP da câmera (conecta ao Pi Zero)
├── leds.py                   # Cliente HTTP dos LEDs (conecta ao Pi Zero)
├── biometric_template.py     # Extração de template LBP + matching cosseno
├── preprocess_veins.py       # Pipeline IR: Black-Hat + CLAHE + gamma
├── interface.py              # Overlay de guia de posicionamento
├── detector_mao.py           # Detector HOG+SVM (mão / não-mão)
├── database.py               # Banco SQLite de usuários
├── treinar_detector.py       # Treinamento do detector SVM
├── capturar_treino_detector.py  # Captura de dataset para treino
├── testar_captura.py         # Teste de captura e pré-processamento
├── launch.sh                 # Launcher automático (Linux)
├── requirements.txt          # Dependências do PC
└── pi_zero/
    ├── server.py             # Servidor Flask (roda no Pi Zero)
    ├── requirements.txt      # Dependências do Pi Zero
    └── install.sh            # Instalador automático para Pi Zero
```

---

## Configurações avançadas (`config.py`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `PI_ZERO_HOST` | `192.168.1.100` | IP do Pi Zero na rede |
| `PI_ZERO_PORT` | `5000` | Porta do servidor Flask |
| `TIMEOUT_CONNECT` | `3` s | Timeout de conexão |
| `TIMEOUT_READ` | `8` s | Timeout de leitura |
| `JPEG_QUALITY` | `85` | Qualidade dos frames transmitidos (0–100) |
| `FRAME_WIDTH` | `1280` | Resolução horizontal da câmera |
| `FRAME_HEIGHT` | `960` | Resolução vertical da câmera |

Para ajustar o threshold de reconhecimento, edite `biometric_template.py`:
```python
MATCH_THRESHOLD = 0.82   # aumentar = mais restrito / diminuir = mais permissivo
```

---

## Segurança e privacidade

- **Nenhuma imagem é salva em disco** — apenas templates biométricos matemáticos
- Os templates LBP são **irreconstruíveis** — não é possível obter a imagem original a partir deles
- O banco de dados (`usuarios.db`) fica apenas no PC, nunca no Pi Zero
- O Pi Zero transmite apenas frames de vídeo pela rede local — não há conexão com internet
