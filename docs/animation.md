# La explicación animada de BERT, BIO y los folds

El script [bert_explained.py](../animations/bert_explained.py) funciona por sí
solo: incluye la escena, ejemplos inventados, dependencias Python y una guía
de lectura en español. No necesita los datos privados ni instalar el harness.

Las palabras se transforman en subpalabras y vectores; partículas muestran
cómo la atención mezcla información y cómo vuelven los gradientes. Los 750
puntos representan notas que se desplazan entre train y test. Las cinco salidas
test terminan en un conjunto de predicciones out-of-fold. La evaluación cambia
las fronteras de un span para mostrar strict y overlap.

## Instalar las dependencias gráficas

Con `uv` instalado, prepara Cairo y Pango. En el Mac:

```bash
brew install cairo pango pkg-config
```

En Debian o Ubuntu:

```bash
sudo apt install build-essential python3-dev pkg-config libcairo2-dev libpango1.0-dev
```

El script usa `Text`, sin fórmulas LaTeX ni assets externos. `uv run --script`
instala Manim 0.21.0 en un entorno de script separado, sin añadir paquetes al
entorno BERT/LLM del proyecto. La primera ejecución puede compilar dependencias
gráficas. El [manual de instalación de Manim](https://docs.manim.community/en/stable/installation/uv.html)
detalla las dependencias de cada plataforma.

## Renderizar

Desde la raíz del repositorio:

```bash
# Vídeo completo a 1080p y 30 fps.
uv run --script animations/bert_explained.py --quality high --fps 30

# Vista previa pequeña, cinco veces más rápida.
uv run --script animations/bert_explained.py \
  --quality low --pace 0.2 --output-dir runs/animations/preview
```

El vídeo completo tiene texto en español y es **sin voz**. `--pace 1`, el valor
predeterminado, deja tiempo para leer. `--pace 0.2` comprueba la animación con
duraciones más cortas; no es la versión de lectura. `--quality` selecciona
480p, 720p o 1080p; `--fps` permite cambiar la tasa de fotogramas.

Los resultados se guardan en `runs/animations/`:

- `BertSinMisterio.mp4`: vídeo de doce capítulos.
- `explicacion.md`: guía completa, incluidos límites y detalles del paper.
- `capitulos.json`: títulos y tiempos de comienzo, en segundos.

Manim también escribe archivos intermedios en ese directorio. `runs/` ya está
ignorado por Git. Puedes copiar solo el script a otra máquina y ejecutarlo con
el mismo comando. En el Mac, abre el resultado con:

```bash
open runs/animations/BertSinMisterio.mp4
```

## Qué explica y de dónde salen las cifras

La animación distingue el encoder preentrenado y la nueva cabeza NER, el
entrenamiento supervisado y el prompting de un LLM. La conversión de spans a
BIO y su reconstrucción corresponden a `align_labels` y `decode_window` de
[methods/bert.py](../methods/bert.py). Las limitaciones del adaptador están en
la [guía de BERT](methods/bert.md); el scorer está explicado en la
[guía de evaluación](evaluation.md).

El protocolo de 750 notas, cinco folds, 600 train/150 test y fine-tuning viene
del manuscrito suministrado, especialmente sus secciones 3.6 y 3.8. El F1 de
BETO se calcula juntando predicciones test, no sobre train. Los folds se asignan
por documento, no por paciente. No disponemos de los IDs, la semilla exacta ni
su implementación completa para afirmar una réplica idéntica.

Los cinco folds corresponden a los baselines BETO/BETO Galen. El sistema de
reglas se compara con las anotaciones corregidas del conjunto de evaluación;
no tiene un entrenamiento de pesos con esos folds. El LLM del paper tampoco
entrena pesos: se evalúa zero-shot y excluye un timeout, con 749 notas puntuadas.

Vectores, pesos de atención y cambios de probabilidades son dibujos
ilustrativos. El script no ejecuta BERT, BETO ni un LLM, y no entrena modelos.
La representación de seis coordenadas simplifica los vectores de 768
dimensiones de BETO. El pequeño grafo de los folds es un icono del modelo; la
arquitectura se explica antes con las doce capas del encoder y la cabeza
compartida. No se incluye el esquema privado ni contenido de notas reales.
