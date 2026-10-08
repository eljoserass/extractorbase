# /// script
# requires-python = ">=3.12"
# dependencies = ["manim==0.21.0"]
# ///
"""A self-contained, silent Spanish lesson about the NER experiment.

Render with uv (isolated script dependencies; no extractor installation needed):
    uv run --script animations/bert_explained.py --quality high --fps 30
    uv run --script animations/bert_explained.py --quality low --pace 0.25

Native prerequisites:
    macOS: brew install cairo pango pkg-config
    Debian/Ubuntu:
        sudo apt update
        sudo apt install build-essential python3-dev pkg-config libcairo2-dev libpango1.0-dev
        pkg-config --modversion cairo pangocairo

No LaTeX, dataset, schema, model downloads or external assets are used. All
examples and probabilities are invented. This draws an explanation; it does
not train a model. The supplied manuscript's sections 3.6 and 3.8 are the
source for its training and evaluation protocol. Exact fold IDs, fold seed,
window-merge algorithm and verbatim LLM prompt were not supplied.

Output: an MP4, a Spanish reading guide and a chapter timestamp file under
--output-dir (default: runs/animations). --pace scales pauses and animations;
1.0 is the reading version. There is no generated voiceover.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Animation,
    ArcBetweenPoints,
    Arrow,
    ChangeDecimalToValue,
    Circle,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    MoveAlongPath,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    TransformFromCopy,
    TransformMatchingShapes,
    VGroup,
    Wait,
    tempconfig,
)

BACKGROUND = "#0B1019"
PANEL = "#1D293D"
INK = "#EDF3FC"
MUTED = "#AEC0D9"
TRAIN = "#57B8FF"
TEST = "#FFC857"
GOLD = "#74D8A8"
GRADIENT = "#E898D5"
CHAPTERS = 12

READING_GUIDE = """# BERT, BIO y los cinco folds

Los ejemplos son inventados. El vídeo no usa ni modifica datos, esquemas o
pesos. Las cifras del protocolo vienen del manuscrito suministrado, secciones
3.3, 3.6–3.8; no disponemos de sus IDs exactos, semilla de folds, código de
fusión de ventanas ni prompt literal.

1. **La tarea.** Reconocer entidades en una nota y devolver su tipo y sus
   posiciones de caracteres. Los baselines BETO y LLM del paper evalúan solo
   entidades; el sistema de reglas también aborda relaciones y negación.
2. **Label Studio y BIO.** Label Studio guarda spans [start, end), no etiquetas
   por token. Nuestro `align_labels` convierte esos spans a BIO al hacer fit:
   B comienza una entidad, I continúa y O queda fuera. El tokenizer aporta
   offsets para alinear subpalabras. `decode_window` reconstruye spans durante
   predict. No hace falta cambiar los JSON originales ni buscar otro conversor.
   Una secuencia BIO plana no puede representar todas las entidades solapadas;
   también puede perder fronteras situadas dentro de una subpalabra. Nuestro
   adaptador prioriza spans largos, ignora fragmentos en bordes de ventanas y
   omite tokens especiales en la loss. Esos límites merecen medirse, pero no
   explican por sí solos la mala salida del clasificador que probamos.
3. **Arquitectura.** BETO es un encoder BERT-base de 12 capas y 768 dimensiones
   por token. Self-attention combina el contexto disponible de la ventana, en
   ambas direcciones. Una cabeza de clasificación compartida aplica una
   transformación a cada vector contextual y produce logits para 2N+1 tags.
   N es el número de tipos de entidad: B e I para cada tipo, más un único O.
   Dos tipos generan cinco clases, no seis. La misma W y el mismo b se aplican
   a todos los tokens en paralelo. Softmax normaliza cada fila; argmax elige
   un tag. Argmax sobre logits elegiría el mismo tag: softmax no cambia cuál
   es el mayor. Nuestro código usa softmax y max para obtener también la
   confianza con la que resuelve candidatos solapados entre ventanas.
   El vídeo sigue dolor/lum/##bar/hoy hasta B-H/I-H/I-H/O. El código agrupa
   los tres primeros tokens y usa sus offsets: start=6, end=18 (exclusivo),
   tipo=HALLAZGO. No hay una segunda cabeza aprendida para start/end;
   esos límites vienen del tokenizer y el tipo viene de los tags seleccionados.
   La arquitectura dibujada es BETO: nuestro checkpoint original es DistilBERT
   multilingüe y tiene seis capas, aunque conserva esta misma idea de encoder
   más clasificador. No estamos dibujando una arquitectura generativa de LLM.
4. **Preentrenamiento no equivale a NER entrenado.** El encoder ya aprendió
   representaciones del idioma. Una nueva cabeza para nuestras etiquetas empieza
   aleatoria; los nombres de los códigos no le enseñan automáticamente su
   significado. No hay un prompt de instrucciones en esta implementación.
5. **Entrenamiento.** Trainer calcula cross-entropy sobre los tags BIO gold.
   Backprop calcula gradientes y AdamW actualiza los parámetros entrenables.
   El F1 del evaluador no es esa loss y no dirige los pesos. Los tags gold de
   las notas test se usan al puntuar, nunca para entrenar ese fold. Las
   probabilidades del vídeo son ilustrativas, no resultados de un experimento.
6. **Nuestra configuración frente al paper.** Nuestro fit por defecto congela
   el encoder y entrena la nueva cabeza durante tres épocas. El artefacto débil
   se ajustó con cinco notas. El paper hace fine-tuning de BETO y BETO Galen:
   cuatro épocas, batch 8, learning rate 3e-5, AdamW, sin warmup, en cada fold.
   Diez notas no garantizan una cabeza útil; inferir con una cabeza nueva sin
   entrenamiento tampoco constituye un baseline zero-shot de NER válido.
7. **Notas largas.** El paper usa ventanas de 512 tokens y stride 128 (solape),
   luego fusiona spans por offsets de caracteres. Nuestro default usa 256/64.
   Las ventanas de una misma nota deben permanecer en el mismo fold. El
   algoritmo exacto de fusión del paper no está descrito con suficiente detalle
   para afirmar que coincide con nuestro filtrado por confianza.
8. **Cinco folds.** Dividen 750 notas curadas en cinco grupos de 150 documentos.
   En cada ronda, cuatro grupos forman train (600) y el quinto test (150).
   Se inicia un modelo independiente desde el checkpoint preentrenado, no
   desde el modelo ya ajustado en el fold anterior. El paper asigna por documento,
   no por paciente: puede haber notas del mismo paciente a ambos lados.
   Su selección busca cubrir etiquetas raras; no afirma duplicar notas para
   entrenar. No describe otro split de validación ni early stopping.
9. **Out-of-fold.** Cada nota obtiene exactamente una predicción de un modelo
   que no usó esa nota para entrenar. Juntan las cinco salidas de 150 y calculan
   los scores sobre esas 750 predicciones. Cada nota participa en train en otras
   cuatro rondas. Son cinco modelos, no uno entrenado secuencialmente cinco veces.
10. **F1.** Strict exige tipo, start y end iguales. Overlap acepta intersección
    positiva con el mismo tipo y da crédito completo, con emparejamiento uno
    a uno. Un fallo de frontera puede ser FP+FN en strict y TP en overlap.
    Micro suma TP/FP/FN de todas las notas y tipos antes de calcular P, R y F1.
    Macro promedia F1 por tipo. Pooling no significa promediar los cinco F1.
    Nuestro scorer usa nervaluate con una adaptación para overlap de crédito
    completo. El paper no detalla desempates ni la política de tipos sin soporte;
    nuestro macro incluye tipos presentes en gold.
11. **El LLM.** El paper sí prueba un LLM zero-shot: Qwen3.5 local recibe los
    códigos y sus descripciones, la nota y una instrucción para devolver JSON
    con citas literales. No entrena pesos ni incluye demostraciones. Nuestro
    fit LLM guarda demostraciones: es few-shot. Su salida vuelve a spans para
    puntuar. Se excluyó un timeout del LLM en el paper (749 notas puntuadas).
12. **Comparación.** Nuestro pequeño fit y holdout actual no reproducen los
    750 documentos y cinco folds originales. grupo1/grupo2 son archivos de
    datos, no los folds del paper. El protocolo y el scorer deben distinguirse:
    tener métricas semejantes no implica haber replicado todo el experimento.
    El F1 de BETO que reportan se calcula en test out-of-fold, no en train.
"""


def text(message: str, size: float = 26, color: str = INK, width: float = 12) -> Text:
    result = Text(message, font_size=size, color=color, line_spacing=0.85)
    if result.width > width:
        result.scale_to_fit_width(width)
    return result


def vector(values: list[float], color: str = TRAIN) -> VGroup:
    """Six visible coordinates stand in for a 768-dimensional representation."""
    return VGroup(
        *(
            Rectangle(
                width=0.32,
                height=0.18,
                stroke_width=0.5,
                stroke_color=MUTED,
                fill_color=color if value >= 0 else GRADIENT,
                fill_opacity=0.2 + min(abs(value), 1) * 0.8,
            )
            for value in values
        )
    ).arrange(DOWN, buff=0.035)


def bracket(start: float, end: float, y: float, color: str) -> VGroup:
    return VGroup(
        Line([start, y, 0], [end, y, 0], color=color, stroke_width=4),
        Line([start, y - 0.1, 0], [start, y + 0.1, 0], color=color),
        Line([end, y - 0.1, 0], [end, y + 0.1, 0], color=color),
    )


def notes(count: int = 150, columns: int = 15, color: str = MUTED) -> VGroup:
    """One point represents one note, including in the 750-note pooled cloud."""
    cloud = VGroup(
        *(
            Dot(
                [index % columns * 0.115, -(index // columns) * 0.115, 0], radius=0.026, color=color
            )
            for index in range(count)
        )
    )
    return cloud.move_to([0, 0, 0])


def model_icon() -> VGroup:
    layers = [
        VGroup(*(Dot([x, y, 0], radius=0.065, color=TRAIN) for y in [0.4, 0, -0.4]))
        for x in [-0.4, 0, 0.4]
    ]
    edges = VGroup(
        *(
            Line(a.get_center(), b.get_center(), color=TRAIN, stroke_width=1, stroke_opacity=0.4)
            for left, right in zip(layers, layers[1:])
            for a in left
            for b in right
        )
    )
    return VGroup(edges, *layers)


class BertSinMisterio(Scene):
    """A moving explanation: words become vectors, gradients flow, notes change folds."""

    def __init__(self, pace: float = 1.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.pace = pace
        self.chapter_times: list[dict[str, float | str | int]] = []

    def play(self, *animations: Animation, run_time: float = 1, **kwargs: object) -> None:
        super().play(*animations, run_time=run_time * self.pace, **kwargs)

    def hold(self, seconds: float = 2) -> None:
        if seconds > 0:
            super().play(Wait(run_time=seconds * self.pace))

    def topic(self, number: int, title: str) -> None:
        self.chapter_times.append({"chapter": number, "title": title, "seconds": self.time})
        print(f"[{number}/{CHAPTERS}] {title}", file=sys.stderr)
        title_object = text(title, 34, width=11.7).to_edge(UP, buff=0.35)
        title_object.to_edge(LEFT, buff=0.65)
        counter = text(f"{number:02d}/{CHAPTERS}", 18, MUTED).to_corner(UP + RIGHT, buff=0.5)
        if number == 1:
            self.title, self.counter = title_object, counter
            self.caption = text("", 24).to_edge(DOWN, buff=0.35)
            self.add(self.caption)
            self.play(FadeIn(self.title), FadeIn(self.counter), run_time=0.6)
        else:
            self.play(
                Transform(self.title, title_object), Transform(self.counter, counter), run_time=0.6
            )

    def say(self, message: str, hold: float = 0) -> None:
        caption = text(message, 24, width=12.5).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(self.caption), FadeIn(caption), run_time=0.35)
        self.caption = caption
        self.hold(hold)

    def flow(self, paths: VGroup, color: str = GOLD, reverse: bool = False) -> None:
        particles = VGroup(*(Dot(radius=0.045, color=color) for _ in paths))
        animations = []
        for particle, path in zip(particles, paths):
            route = path.copy()
            if reverse:
                route.reverse_points()
            particle.move_to(route.get_start())
            animations.append(MoveAlongPath(particle, route))
        self.add(particles)
        self.play(LaggedStart(*animations, lag_ratio=0.08), run_time=1.3)
        # AnimationGroup can promote its children to scene roots; remove both levels.
        self.remove(particles, *particles)

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        self.spans_to_bio()
        self.contextual_encoder()
        self.classifier_and_training()
        self.cross_validation()
        self.scoring()
        self.llm_and_conclusion()

    def spans_to_bio(self) -> None:
        self.topic(1, "La tarea empieza en el texto")
        words = VGroup(*(text(word, 43) for word in ["Tiene", "dolor", "lumbar", "hoy."]))
        words.arrange(RIGHT, buff=0.3).move_to(UP * 0.9)
        self.play(LaggedStart(*(FadeIn(word, shift=UP * 0.2) for word in words), lag_ratio=0.15))
        self.say("Queremos encontrar menciones y devolver su tipo y sus posiciones.")
        self.span = bracket(words[1].get_left()[0], words[2].get_right()[0], -0.05, GOLD)
        label = text("HALLAZGO", 27, GOLD).move_to(DOWN * 0.55)
        self.play(
            words[1].animate.set_color(GOLD),
            words[2].animate.set_color(GOLD),
            Create(self.span),
            FadeIn(label),
        )
        offsets = text("Label Studio: start=6, end=18", 26, MUTED).move_to(DOWN * 1.25)
        self.play(FadeIn(offsets))
        self.say(
            "Label Studio guarda un span de caracteres: [6, 18).\n"
            "Todos los textos y valores de esta animación son inventados.",
            hold=4,
        )

        self.topic(2, "El adaptador convierte spans en BIO")
        self.tokens = VGroup(
            *(text(word, 29) for word in ["Tiene", "dolor", "lum", "##bar", "hoy", "."])
        )
        self.tokens.arrange(RIGHT, buff=0.65).move_to(UP * 1.5)
        self.play(
            TransformMatchingShapes(words, self.tokens),
            Transform(
                self.span,
                bracket(self.tokens[1].get_left()[0], self.tokens[3].get_right()[0], 0.4, GOLD),
            ),
            FadeOut(offsets),
            run_time=1.5,
        )
        self.say(
            "Un tokenizer puede dividir una palabra en subpalabras.\n"
            "Este corte es ilustrativo: los offsets enlazan tokens y texto original.",
            hold=2,
        )
        token_offsets = [(0, 5), (6, 11), (12, 15), (15, 18), (19, 22), (22, 23)]
        offset_labels = VGroup(
            *(
                text(f"[{start}, {end})", 19, MUTED).move_to(token.get_center() + DOWN * 0.55)
                for token, (start, end) in zip(self.tokens, token_offsets)
            )
        )
        self.play(
            LaggedStart(*(FadeIn(item) for item in offset_labels), lag_ratio=0.12),
            label.animate.scale(0.7).move_to(DOWN * 1.55),
        )
        tags = ["O", "B", "I", "I", "O", "O"]
        self.bio_tags = VGroup(
            *(
                text(tag, 40, GOLD if tag != "O" else MUTED).move_to(token.get_center() + DOWN * 2)
                for token, tag in zip(self.tokens, tags)
            )
        )
        paths = VGroup(
            *(
                Line(label.get_center(), tag.get_center(), stroke_opacity=0)
                for tag in list(self.bio_tags)[1:4]
            )
        )
        self.flow(paths)
        self.play(
            LaggedStart(*(FadeIn(tag, shift=DOWN * 0.25) for tag in self.bio_tags), lag_ratio=0.15)
        )
        self.say(
            "B empieza HALLAZGO; I lo continúa; O queda fuera.\n"
            "align_labels ya hace esto en nuestro código. No hay que cambiar los JSON.",
            hold=4,
        )
        self.play(Indicate(self.bio_tags[1], color=GOLD), Indicate(self.bio_tags[2:4], color=GOLD))
        self.say(
            "Una secuencia BIO plana solo asigna un tag por token.\n"
            "Solapes y fronteras dentro de subpalabras pueden perder información.",
            hold=4,
        )
        self.play(
            FadeOut(self.span),
            FadeOut(label),
            FadeOut(offset_labels),
            self.bio_tags.animate.scale(0.6).to_edge(DOWN, buff=1.1),
            self.tokens.animate.move_to(UP * 2.2),
        )

    def contextual_encoder(self) -> None:
        self.topic(3, "Cada token se transforma en un vector")
        self.say("Token + posición → vector. BETO usa 768 dimensiones; dibujamos seis.")
        self.embeddings = VGroup(
            *(
                vector([math.sin(i * 1.7 + j) for j in range(6)]).move_to(
                    token.get_center() + DOWN * 1.8
                )
                for i, token in enumerate(self.tokens)
            )
        )
        arrows = VGroup(
            *(
                Arrow(
                    token.get_bottom(), embedding.get_top(), buff=0.15, color=TRAIN, stroke_width=2
                )
                for token, embedding in zip(self.tokens, self.embeddings)
            )
        )
        self.play(
            LaggedStart(
                *(
                    TransformFromCopy(token, embedding)
                    for token, embedding in zip(self.tokens, self.embeddings)
                ),
                lag_ratio=0.13,
            ),
            Create(arrows),
            run_time=2,
        )
        self.say(
            "El modelo suma un embedding del token y uno de su posición.\n"
            "BETO usa 768 dimensiones; aquí solo dibujamos seis para verlas.",
            hold=3,
        )
        positions = VGroup(
            *(
                vector([math.cos(i + j * 0.8) for j in range(6)], TEST).move_to(
                    embedding.get_center() + RIGHT * 0.5
                )
                for i, embedding in enumerate(self.embeddings)
            )
        )
        self.play(FadeIn(positions, shift=UP * 0.2))
        self.play(
            *(
                Transform(
                    embedding,
                    vector(
                        [
                            math.sin(i * 1.7 + j) * 0.7 + math.cos(i + j * 0.8) * 0.3
                            for j in range(6)
                        ]
                    ).move_to(embedding),
                )
                for i, embedding in enumerate(self.embeddings)
            ),
            FadeOut(positions, shift=LEFT * 0.4),
            run_time=1.5,
        )
        self.play(FadeOut(arrows), FadeOut(self.bio_tags))

        self.topic(4, "La atención mezcla información del contexto")
        self.say("Q consulta K; los pesos de atención deciden cuánto aporta cada V.")
        query = Dot(LEFT * 2 + DOWN * 1.55, color=TEST, radius=0.12)
        query_label = text("Q de “dolor”", 24, TEST).next_to(query, DOWN, buff=0.25)
        weights = [0.05, 0.15, 0.35, 0.25, 0.10, 0.10]
        beams = VGroup(
            *(
                Line(
                    embedding.get_bottom(),
                    query.get_center(),
                    color=TRAIN,
                    stroke_width=1 + weight * 12,
                    stroke_opacity=0.3 + weight,
                )
                for embedding, weight in zip(self.embeddings, weights)
            )
        )
        weight_labels = VGroup(
            *(
                text(f"{weight:.2f}", 19, TEST).next_to(embedding, DOWN, buff=0.1)
                for embedding, weight in zip(self.embeddings, weights)
            )
        )
        self.play(TransformFromCopy(self.embeddings[1], query), FadeIn(query_label), Create(beams))
        self.say(
            "La query Q del token consulta las keys K: salen pesos de atención.\n"
            "Con esos pesos se mezclan los vectores V. Estos pesos son ilustrativos.",
            hold=2,
        )
        self.play(LaggedStart(*(FadeIn(item) for item in weight_labels), lag_ratio=0.1))
        self.flow(beams)
        self.flow(beams)
        context = vector([0.6, -0.3, 0.8, 0.1, -0.5, 0.4], GOLD).move_to(RIGHT * 3 + DOWN * 1.3)
        context_label = text("nuevo vector contextual", 23, GOLD).next_to(context, RIGHT, buff=0.3)
        context_label.scale_to_fit_width(3.1)
        self.play(TransformFromCopy(query, context), FadeIn(context_label))
        self.say(
            "El vector de “dolor” cambia al mirar los otros tokens.\n"
            "En BERT, cada token puede consultar ambos lados de la ventana.",
            hold=4,
        )
        self.play(
            FadeOut(beams),
            FadeOut(weight_labels),
            FadeOut(query),
            FadeOut(query_label),
            FadeOut(context_label),
            FadeOut(context),
        )

        self.topic(5, "BETO repite atención + feed-forward en 12 capas")
        self.say(
            "Cada capa mezcla contexto y transforma los vectores.\n"
            "Las conexiones residuales y la normalización estabilizan este recorrido."
        )
        self.play(
            self.embeddings.animate.scale(0.65).move_to(UP * 1.65),
            self.tokens.animate.scale(0.8).move_to(UP * 2.3),
        )
        layers = (
            VGroup(
                *(
                    RoundedRectangle(
                        width=6.8,
                        height=0.13,
                        corner_radius=0.035,
                        color=TRAIN if i % 2 == 0 else GRADIENT,
                        fill_opacity=0.15,
                        stroke_width=1.2,
                    )
                    for i in range(12)
                )
            )
            .arrange(DOWN, buff=0.1)
            .move_to(DOWN * 0.1)
        )
        layer_count = text("12 capas\n768 dimensiones", 25, MUTED).next_to(layers, LEFT, buff=0.4)
        self.play(
            LaggedStart(*(Create(layer) for layer in layers), lag_ratio=0.08), FadeIn(layer_count)
        )
        packets = self.embeddings.copy()
        self.add(packets)
        for layer in layers:
            self.play(
                packets.animate.move_to(layer.get_center()),
                layer.animate.set_fill(GOLD, opacity=0.5),
                run_time=0.16,
            )
        output = VGroup(
            *(
                vector([math.sin(i + j * 0.7) for j in range(6)], GOLD)
                .scale(0.65)
                .move_to([embedding.get_x(), -1.9, 0])
                for i, embedding in enumerate(self.embeddings)
            )
        )
        self.play(ReplacementTransform(packets, output), run_time=1)
        self.say(
            "Atención + red feed-forward, con conexiones residuales y normalización.\n"
            "El resultado conserva un vector contextual por token.",
            hold=3,
        )
        self.say(
            "Eso describe BETO. Nuestro DistilBERT multilingüe tiene seis capas.\n"
            "Las notas largas se procesan por ventanas, no todas de una vez.",
            hold=3,
        )
        self.encoder_icon = VGroup(layers, layer_count)
        self.h = output[1].copy()
        self.play(
            self.encoder_icon.animate.scale(0.23).move_to(LEFT * 5 + UP * 2.25),
            FadeOut(self.tokens),
            FadeOut(self.embeddings),
            FadeOut(output),
            self.h.animate.scale(2).move_to(LEFT * 4),
            run_time=1.5,
        )

    def tags_and_spans(self) -> None:
        """Expand the label space, classify subwords, then recover character spans."""
        self.play(FadeOut(self.h), FadeOut(self.encoder_icon))
        count = text("N = 1 tipo de entidad", 30, GOLD).move_to(UP * 2.3)
        finding = text("HALLAZGO", 28, GOLD).move_to(LEFT * 1.2 + UP * 1.25)
        outside = text("O", 34, MUTED).move_to(LEFT * 5 + DOWN * 0.3)
        begin = text("B-HALLAZGO", 25, GOLD).move_to(LEFT * 2.5 + DOWN * 0.3)
        inside = text("I-HALLAZGO", 25, GOLD).move_to(RIGHT * 0.2 + DOWN * 0.3)
        branches = VGroup(
            Arrow(finding.get_bottom(), begin.get_top(), color=GOLD, buff=0.15),
            Arrow(finding.get_bottom(), inside.get_top(), color=GOLD, buff=0.15),
        )
        formula = text("1 + 2 × 1 = 3 clases", 34, TEST).move_to(DOWN * 1.8)
        self.say(
            "N cuenta tipos de entidad. Cada tipo aporta B e I; O significa fuera de toda entidad."
        )
        self.play(FadeIn(count), FadeIn(finding), FadeIn(outside))
        self.play(
            Create(branches), TransformFromCopy(finding, begin), TransformFromCopy(finding, inside)
        )
        self.play(FadeIn(formula))
        self.hold(3)

        medicine = text("MEDICAMENTO", 26, GRADIENT).move_to(RIGHT * 4 + UP * 1.25)
        med_begin = text("B-MEDICAMENTO", 20, GRADIENT).move_to(RIGHT * 2.7 + DOWN * 0.3)
        med_inside = text("I-MEDICAMENTO", 20, GRADIENT).move_to(RIGHT * 5.2 + DOWN * 0.3)
        med_branches = VGroup(
            Arrow(medicine.get_bottom(), med_begin.get_top(), color=GRADIENT, buff=0.15),
            Arrow(medicine.get_bottom(), med_inside.get_top(), color=GRADIENT, buff=0.15),
        )
        self.play(
            Transform(count, text("N = 2 tipos de entidad", 30, GOLD).move_to(count)),
            FadeIn(medicine),
        )
        self.play(
            Create(med_branches),
            TransformFromCopy(medicine, med_begin),
            TransformFromCopy(medicine, med_inside),
            Transform(formula, text("1 + 2 × 2 = 5 clases", 34, TEST).move_to(formula)),
            Indicate(outside),
        )
        self.say("O es una sola clase compartida: no existe un O diferente por tipo.", hold=4)
        self.play(Transform(formula, text("N tipos → 2N + 1 clases", 34, TEST).move_to(formula)))
        self.hold(3)

        # Keep the five classes visible while their layout becomes a probability matrix.
        class_labels = VGroup(outside, begin, inside, med_begin, med_inside)
        short_tags = ["O", "B-H", "I-H", "B-M", "I-M"]
        class_colors = [MUTED, GOLD, GOLD, GRADIENT, GRADIENT]
        headings = [
            text(tag, 23, color).move_to([-0.8 + 1.3 * index, 1.35, 0])
            for index, (tag, color) in enumerate(zip(short_tags, class_colors))
        ]
        self.play(
            *(Transform(label, heading) for label, heading in zip(class_labels, headings)),
            FadeOut(finding),
            FadeOut(medicine),
            FadeOut(branches),
            FadeOut(med_branches),
            FadeOut(count),
            FadeOut(formula),
        )
        shared = RoundedRectangle(width=1.25, height=3.2, corner_radius=0.2, color=TRAIN).move_to(
            LEFT * 3
        )
        weights = text("W, b", 27, TRAIN).move_to(LEFT * 3 + UP * 1.1)
        shared_label = text("Una misma cabeza: zᵢ = W·hᵢ + b", 29, TRAIN).move_to(UP * 2.4)
        probability_label = text("softmax por fila → argmax", 23, MUTED).move_to(
            RIGHT * 2.2 + UP * 1.95
        )
        tokens = VGroup(
            *(
                text(word, 25).move_to([-5.55, 0.55 - i * 0.65, 0])
                for i, word in enumerate(["dolor", "lum", "##bar", "hoy"])
            )
        )
        vectors = VGroup(
            *(
                vector([0.2 + i * 0.15, -0.5, 0.8 - i * 0.1, 0.3, -0.2, 0.6])
                .scale(0.32)
                .move_to([-4.35, token.get_y(), 0])
                for i, token in enumerate(tokens)
            )
        )
        routes = VGroup(
            *(
                Line(v.get_right(), [-1.65, v.get_y(), 0], color=TRAIN, stroke_opacity=0.45)
                for v in vectors
            )
        )
        self.say("H = HALLAZGO; M = MEDICAMENTO. Cada fila corresponde a un token contextualizado.")
        self.play(
            FadeIn(tokens),
            FadeIn(vectors),
            Create(shared),
            FadeIn(weights),
            FadeIn(shared_label),
            FadeIn(probability_label),
            Create(routes),
        )
        self.flow(routes)
        probabilities = [
            [0.04, 0.88, 0.04, 0.02, 0.02],
            [0.03, 0.04, 0.89, 0.02, 0.02],
            [0.02, 0.03, 0.91, 0.02, 0.02],
            [0.92, 0.02, 0.02, 0.02, 0.02],
        ]
        cells = VGroup()
        winners = VGroup()
        selections = VGroup()
        for row, values in enumerate(probabilities):
            winner = values.index(max(values))
            for column, probability in enumerate(values):
                cell = VGroup(
                    Rectangle(
                        width=1.03,
                        height=0.48,
                        stroke_width=0,
                        fill_color=GOLD,
                        fill_opacity=probability * 0.6,
                    ),
                    text(f"{probability:.2f}", 22),
                ).move_to([-0.8 + 1.3 * column, 0.55 - row * 0.65, 0])
                cells.add(cell)
                if column == winner:
                    selections.add(
                        Rectangle(width=1.09, height=0.54, color=TEST, stroke_width=2).move_to(cell)
                    )
            winners.add(text(short_tags[winner], 23, GOLD).move_to([5.7, 0.55 - row * 0.65, 0]))
        self.play(LaggedStart(*(FadeIn(cell) for cell in cells), lag_ratio=0.025), run_time=2)
        self.say(
            "Softmax reparte probabilidad entre las cinco clases: cada fila suma 1.\n"
            "Son cifras inventadas de un clasificador ya entrenado.",
            hold=4,
        )
        self.play(
            Create(selections), LaggedStart(*(FadeIn(winner) for winner in winners), lag_ratio=0.15)
        )
        self.say(
            "Argmax elige un tag por token. W y b se comparten; "
            "las filas se calculan en paralelo.\n"
            "No se genera el siguiente token como en un LLM.",
            hold=5,
        )

        # The selected tags survive the network diagram and become a character span.
        self.play(
            FadeOut(shared),
            FadeOut(weights),
            FadeOut(shared_label),
            FadeOut(probability_label),
            FadeOut(vectors),
            FadeOut(routes),
            FadeOut(cells),
            FadeOut(selections),
            FadeOut(class_labels),
            *(token.animate.move_to([-4.5 + 3 * i, 1.4, 0]) for i, token in enumerate(tokens)),
            *(winner.animate.move_to([-4.5 + 3 * i, 0.65, 0]) for i, winner in enumerate(winners)),
            run_time=1.5,
        )
        offsets = VGroup(
            *(
                text(offset, 24, MUTED).move_to([-4.5 + 3 * i, -0.1, 0])
                for i, offset in enumerate(["[6, 11)", "[12, 15)", "[15, 18)", "[19, 22)"])
            )
        )
        origin = text("Offsets del tokenizer, conocidos antes de predecir", 27, MUTED).move_to(
            UP * 2.4
        )
        self.play(FadeIn(offsets), FadeIn(origin))
        self.say(
            "El código agrupa B-H seguido de I-H, I-H. O queda fuera.\n"
            "Los números son posiciones de caracteres en «Tiene dolor lumbar hoy.».",
            hold=5,
        )
        span = bracket(-5.25, 2.25, -0.75, GOLD)
        self.play(
            Create(span),
            tokens[3].animate.set_opacity(0.3),
            winners[3].animate.set_opacity(0.3),
            offsets[3].animate.set_opacity(0.3),
        )
        start = text("start = 6", 29, TEST).move_to(LEFT * 3 + DOWN * 1.35)
        end = text("end = 18", 29, TEST).move_to(RIGHT * 0.3 + DOWN * 1.35)
        entity_type = text("tipo = HALLAZGO", 26, GOLD).move_to(RIGHT * 4 + DOWN * 1.35)
        self.play(
            TransformFromCopy(offsets[0], start),
            TransformFromCopy(offsets[2], end),
            TransformFromCopy(winners[0], entity_type),
        )
        extracted = text("texto[6:18] = «dolor lumbar»", 30, GOLD).move_to(DOWN * 2.2)
        self.play(FadeIn(extracted))
        self.say(
            "start viene del primer token; end, del último (exclusivo). El tipo viene del tag.\n"
            "Aquí no hay otra cabeza entrenada para predecir start y end.",
            hold=6,
        )
        self.play(
            *(
                FadeOut(obj)
                for obj in [
                    tokens,
                    winners,
                    offsets,
                    origin,
                    span,
                    start,
                    end,
                    entity_type,
                    extracted,
                ]
            )
        )
        self.remove(cells, *cells, winners, *winners, class_labels, *class_labels)
        self.play(FadeIn(self.h), FadeIn(self.encoder_icon))

    def classifier_and_training(self) -> None:
        self.topic(6, "La cabeza convierte cada vector en tags")
        self.tags_and_spans()
        self.say("Para ver el entrenamiento de cerca, volvemos a N = 1: O, B-HALLAZGO, I-HALLAZGO.")
        h_label = text("h: vector de “dolor”", 24, GOLD).move_to(LEFT * 4.4 + DOWN * 1.65)
        input_nodes = VGroup(
            *(Dot([-3.4, 1.25 - i * 0.5, 0], radius=0.07, color=GOLD) for i in range(6))
        )
        output_nodes = VGroup(
            *(Dot([0.6, 1.25 - i * 1.25, 0], radius=0.1, color=TRAIN) for i in range(3))
        )
        edges = VGroup(
            *(
                Line(
                    a.get_center(),
                    b.get_center(),
                    color=TRAIN,
                    stroke_opacity=0.25 + (index % 4) * 0.12,
                    stroke_width=1.4,
                )
                for index, (a, b) in enumerate((a, b) for a in input_nodes for b in output_nodes)
            )
        )
        tag_labels = VGroup(
            *(
                text(tag, 24, GOLD if i == 1 else INK).move_to([3.1, 1.6 - i * 1.25, 0])
                for i, tag in enumerate(["O", "B-HALLAZGO", "I-HALLAZGO"])
            )
        )
        probabilities = [0.36, 0.31, 0.33]
        bars = VGroup(
            *(
                Rectangle(
                    width=2.8 * probability,
                    height=0.13,
                    stroke_width=0,
                    fill_color=TRAIN,
                    fill_opacity=0.8,
                ).move_to([1.5 + 1.4 * probability, 1.12 - i * 1.25, 0])
                for i, probability in enumerate(probabilities)
            )
        )
        probability_labels = VGroup(
            *(
                DecimalNumber(probability, mob_class=Text, font_size=20, color=MUTED).move_to(
                    [5.1, 1.12 - i * 1.25, 0]
                )
                for i, probability in enumerate(probabilities)
            )
        )
        head_label = text("W·h + b → softmax", 26, TRAIN).move_to(UP * 2.35)
        self.play(
            FadeIn(h_label),
            TransformFromCopy(self.h, input_nodes),
            Create(edges),
            FadeIn(output_nodes),
            FadeIn(head_label),
            run_time=1.5,
        )
        self.flow(edges[::3])
        self.play(
            LaggedStart(*(FadeIn(label) for label in tag_labels), lag_ratio=0.15),
            FadeIn(bars),
            FadeIn(probability_labels),
        )
        self.say(
            "Una misma cabeza se aplica a cada token: logits → probabilidades → tag.\n"
            "N tipos necesitan 2N+1 salidas. Aquí usamos un tipo y tres salidas.",
            hold=4,
        )
        self.say(
            "El encoder ya está preentrenado; nuestra cabeza nueva empieza aleatoria.\n"
            "Los nombres de las etiquetas no funcionan como instrucciones o prompts.",
            hold=4,
        )

        self.topic(7, "Fit: gold → loss → gradientes → AdamW")
        self.say(
            "La etiqueta gold indica qué probabilidad tiene que subir durante el entrenamiento."
        )
        target = text("Gold: B-HALLAZGO", 24, GOLD).move_to(LEFT * 4.4 + DOWN * 2.2)
        loss_probability = DecimalNumber(0.31, mob_class=Text, font_size=26, color=TEST)
        loss_value = DecimalNumber(-math.log(0.31), mob_class=Text, font_size=26, color=TEST)

        def update_loss(number: DecimalNumber) -> None:
            number.set_value(-math.log(loss_probability.get_value()))

        loss_value.add_updater(update_loss)
        loss = VGroup(
            text("loss = −log(", 26, TEST), loss_probability, text(") =", 26, TEST), loss_value
        ).arrange(RIGHT, buff=0.06)
        loss.move_to(RIGHT * 1.7 + DOWN * 2.3)
        gold_arrow = Arrow(target.get_right(), loss.get_left(), color=GOLD, buff=0.1)
        self.play(FadeIn(target), FadeIn(loss), Create(gold_arrow))
        self.say(
            "Cross-entropy penaliza la probabilidad baja del tag correcto.\n"
            "Trainer calcula esta loss y backprop; AdamW actualiza los pesos.",
            hold=3,
        )
        for probability in [0.52, 0.73, 0.88]:
            self.flow(edges[1::3], color=GRADIENT, reverse=True)
            next_probabilities = [(1 - probability) * 0.53, probability, (1 - probability) * 0.47]
            updated_bars = VGroup(
                *(
                    Rectangle(
                        width=2.8 * p,
                        height=0.13,
                        stroke_width=0,
                        fill_color=GOLD if i == 1 else TRAIN,
                        fill_opacity=0.8,
                    ).move_to([1.5 + 1.4 * p, 1.12 - i * 1.25, 0])
                    for i, p in enumerate(next_probabilities)
                )
            )
            self.play(
                Transform(bars, updated_bars),
                *(
                    ChangeDecimalToValue(label, p)
                    for label, p in zip(probability_labels, next_probabilities)
                ),
                ChangeDecimalToValue(loss_probability, probability),
                edges.animate.set_color(GRADIENT),
                Indicate(output_nodes[1], color=GOLD),
                run_time=1.2,
            )
            self.hold(0.5)
        self.say(
            "Estos cambios de probabilidad son una simulación didáctica.\n"
            "El F1 de test no dirige los pesos: se calcula después de predecir.",
            hold=4,
        )

        self.topic(8, "Congelar el encoder y hacer fine-tuning son distintos")
        self.say("Podemos actualizar solo la cabeza o también el encoder preentrenado.")
        self.play(self.encoder_icon.animate.set_color(MUTED), Indicate(edges, color=GRADIENT))
        freeze = text("encoder congelado", 22, MUTED).next_to(self.encoder_icon, DOWN, buff=0.2)
        self.play(FadeIn(freeze))
        self.say(
            "Nuestro default entrena solo la cabeza y congela el encoder.\n"
            "El artefacto que probamos se ajustó con cinco notas, durante tres épocas.",
            hold=4,
        )
        encoder_route = VGroup(
            Line(input_nodes[0].get_center(), self.encoder_icon.get_center(), color=GRADIENT)
        )
        self.flow(encoder_route, GRADIENT)
        self.play(self.encoder_icon.animate.set_color(GRADIENT), FadeOut(freeze))
        self.say(
            "El paper hace fine-tuning de BETO/BETO Galen: actualiza el modelo.\n"
            "4 épocas · batch 8 · lr 3e-5 · AdamW · sin warmup.",
            hold=5,
        )
        self.say(
            "Una cabeza aleatoria no es un extractor médico zero-shot.\n"
            "Un pequeño fit tampoco reproduce el entrenamiento del paper.",
            hold=3,
        )
        self.play(
            FadeOut(self.h),
            FadeOut(h_label),
            FadeOut(input_nodes),
            FadeOut(output_nodes),
            FadeOut(edges),
            FadeOut(tag_labels),
            FadeOut(bars),
            FadeOut(probability_labels),
            FadeOut(head_label),
            FadeOut(target),
            FadeOut(loss),
            FadeOut(gold_arrow),
            FadeOut(self.encoder_icon),
        )
        self.remove(probability_labels, *probability_labels, loss, *loss)

    def cross_validation(self) -> None:
        self.topic(9, "750 notas: cada fold usa 600 train y 150 test")
        self.say("Cada punto representa una nota. Formamos cinco grupos de 150.")
        groups = VGroup(*(notes().move_to([-5.1 + 2.55 * i, 1.15, 0]) for i in range(5)))
        source_positions = [group.get_center().copy() for group in groups]
        group_labels = VGroup(
            *(
                text(f"Grupo {i + 1} · 150", 23, MUTED).next_to(group, UP, buff=0.25)
                for i, group in enumerate(groups)
            )
        )
        self.play(
            LaggedStart(*(FadeIn(group, shift=UP * 0.2) for group in groups), lag_ratio=0.15),
            FadeIn(group_labels),
            run_time=2,
        )
        self.say(
            "750 notas curadas se dividen en cinco grupos de 150.\n"
            "Cada punto es una nota. grupo1/grupo2 del repositorio no son estos folds.",
            hold=4,
        )
        self.say(
            "La selección busca cubrir etiquetas raras; no implica duplicar notas.\n"
            "Folds por documento: puede haber pacientes presentes en ambos lados.",
            hold=4,
        )
        self.play(FadeOut(group_labels))
        train_label = text("TRAIN: 600", 27, TRAIN).move_to(LEFT * 3.8 + UP * 0.55)
        test_label = text("TEST: 150", 27, TEST).move_to(RIGHT * 4.7 + UP * 0.95)
        model = model_icon().move_to(RIGHT * 1.3 + DOWN * 1.05)
        model_label = text("BETO₁", 24).next_to(model, DOWN, buff=0.2)
        fit_route = VGroup(Line([-1.2, -1.05, 0], model.get_left(), color=TRAIN))
        test_route = VGroup(ArcBetweenPoints([4.7, -0.75, 0], model.get_right(), angle=0.45))
        self.play(FadeIn(model), FadeIn(model_label))
        ledger = VGroup(
            *(
                Circle(radius=0.13, color=MUTED, stroke_width=2).move_to([-4 + 2 * i, 2.35, 0])
                for i in range(5)
            )
        )
        self.play(Create(ledger))
        self.heldout_predictions = VGroup()
        destinations = [[-5.1, -0.5, 0], [-2.6, -0.5, 0], [-5.1, -1.9, 0], [-2.6, -1.9, 0]]
        for fold in range(5):
            if fold:
                self.say(
                    f"Fold {fold + 1}: reiniciamos desde el checkpoint preentrenado.\n"
                    "Cada ronda entrena un modelo independiente."
                )
                self.play(
                    *(
                        group.animate.move_to(position).set_color(MUTED)
                        for group, position in zip(groups, source_positions)
                    ),
                    model.animate.set_color(MUTED),
                    FadeOut(train_label),
                    FadeOut(test_label),
                    run_time=0.7,
                )
            training = [group for i, group in enumerate(groups) if i != fold]
            self.play(
                *(
                    group.animate.move_to(destination).set_color(TRAIN)
                    for group, destination in zip(training, destinations)
                ),
                groups[fold].animate.move_to([4.7, 0, 0]).set_color(TEST),
                FadeIn(train_label),
                FadeIn(test_label),
                Transform(model_label, text(f"BETO{fold + 1}", 24).move_to(model_label)),
                run_time=1.2,
            )
            if fold == 0:
                self.say(
                    "Cuatro grupos enseñan al modelo: 4 × 150 = 600 notas.\n"
                    "El quinto queda fuera del entrenamiento.",
                    hold=3,
                )
            self.flow(fit_route, TRAIN)
            self.play(model.animate.set_color(GRADIENT), run_time=0.5)
            if fold == 0:
                self.say(
                    "Después el modelo recibe solo el texto de las 150 notas test.\n"
                    "Sus etiquetas gold se reservan para evaluar.",
                    hold=3,
                )
            self.flow(test_route, TEST)
            prediction = groups[fold].copy().set_color(GOLD)
            self.add(prediction)
            self.play(
                prediction.animate.scale(0.36).move_to(ledger[fold].get_center()),
                ledger[fold].animate.set_color(GOLD),
                run_time=1.1,
            )
            self.heldout_predictions.add(prediction)
            self.hold(0.7)
        self.say(
            "Cada nota es test una vez y train en las otras cuatro rondas.\n"
            "No entrenamos un mismo modelo cinco veces seguidas.",
            hold=4,
        )
        self.play(
            FadeOut(groups),
            FadeOut(train_label),
            FadeOut(test_label),
            FadeOut(model),
            FadeOut(model_label),
            FadeOut(ledger),
        )

        self.topic(10, "Juntan las 750 predicciones out-of-fold")
        self.say("Reunimos solo las predicciones de test de los cinco modelos.")
        pooled = notes(750, 30, GOLD).move_to(LEFT * 2)
        for i, prediction in enumerate(self.heldout_predictions):
            target = pooled[i * 150 : (i + 1) * 150]
            self.play(Transform(prediction, target), run_time=0.65)
        total = text("750", 48, GOLD).move_to(RIGHT * 3.6 + UP * 1)
        pooled_label = text("predicciones test", 24, GOLD).next_to(total, DOWN, buff=0.3)
        self.play(FadeIn(total), FadeIn(pooled_label))
        self.say(
            "Cada punto tiene una predicción de un modelo que no entrenó con esa nota.\n"
            "Por eso el F1 publicado de BETO es test out-of-fold, no train.",
            hold=5,
        )
        counts = text("Σ TP    Σ FP    Σ FN\n↓\nPrecision · Recall · F1 micro", 25, TEST)
        counts.move_to(RIGHT * 3.6 + DOWN * 1.2)
        self.play(FadeIn(counts))
        self.say(
            "Para micro suman los conteos de todas las notas antes de calcular F1.\n"
            "Esto no es la media simple de los cinco F1.",
            hold=4,
        )
        self.say(
            "Notas largas: ventanas de 512 tokens con 128 de solape en el paper.\n"
            "Todas las ventanas de una nota deben permanecer en el mismo fold.",
            hold=4,
        )
        self.play(
            FadeOut(self.heldout_predictions),
            FadeOut(total),
            FadeOut(pooled_label),
            FadeOut(counts),
        )

    def scoring(self) -> None:
        self.topic(11, "Se evalúan spans: strict y overlap")
        self.say("Las predicciones vuelven a posiciones de caracteres para compararlas con gold.")
        note = text("Tiene dolor lumbar hoy.", 39).move_to(UP * 2)
        self.play(FadeIn(note))
        scale = 0.43

        def x(offset: float) -> float:
            return (offset - 11.5) * scale

        ruler = VGroup(*(Line([x(i), 0.95, 0], [x(i), 1.05, 0], color=MUTED) for i in range(24)))
        ruler.add(Line([x(0), 1, 0], [x(23), 1, 0], color=MUTED))
        ticks = VGroup(
            *(text(str(i), 20, MUTED).move_to([x(i), 1.35, 0]) for i in [0, 6, 11, 18, 23])
        )
        gold = bracket(x(6), x(18), 0.45, GOLD)
        prediction = bracket(x(6), x(11), -0.05, TEST)
        labels = VGroup(
            text("gold: HALLAZGO", 22, GOLD).move_to(LEFT * 4.4 + DOWN * 0.65),
            text("pred: HALLAZGO", 22, TEST).move_to(RIGHT * 3.8 + DOWN * 0.65),
        )
        self.play(Create(ruler), FadeIn(ticks), Create(gold), Create(prediction), FadeIn(labels))
        self.say(
            "decode_window convierte BIO otra vez en offsets de caracteres.\n"
            "Gold [6,18); predicción [6,11): encontró “dolor” y omitió “lumbar”.",
            hold=4,
        )
        mode = text("STRICT", 28, TRAIN).move_to(LEFT * 4.5 + DOWN * 1.65)
        counts = text("TP=0    FP=1    FN=1", 29).move_to(RIGHT * 0.8 + DOWN * 1.65)
        score = text("F1 = 0", 38, TRAIN).move_to(DOWN * 2.5)
        self.play(FadeIn(mode), FadeIn(counts), FadeIn(score))
        self.say(
            "Strict exige el mismo tipo y las mismas dos fronteras.\n"
            "Esta predicción cuenta como un falso positivo y deja un falso negativo.",
            hold=4,
        )
        self.play(
            Transform(prediction, bracket(x(6), x(18), -0.05, TEST)),
            Transform(counts, text("TP=1    FP=0    FN=0", 29).move_to(counts)),
            Transform(score, text("F1 = 1", 38, GOLD).move_to(score)),
            run_time=1.6,
        )
        self.say("Si el final coincide, strict también la acepta.", hold=3)
        self.play(
            Transform(prediction, bracket(x(6), x(11), -0.05, TEST)),
            Transform(mode, text("OVERLAP", 28, TEST).move_to(mode)),
            run_time=1.2,
        )
        overlap = Rectangle(
            width=x(11) - x(6), height=0.75, stroke_width=0, fill_color=GOLD, fill_opacity=0.2
        ).move_to([(x(6) + x(11)) / 2, 0.2, 0])
        self.play(FadeIn(overlap), Indicate(prediction, color=GOLD))
        self.say(
            "Overlap admite intersección positiva y el mismo tipo: crédito completo.\n"
            "Las parejas son uno a uno; aquí sigue siendo TP=1 y F1=1.",
            hold=4,
        )
        formula = text("P = TP/(TP+FP)     R = TP/(TP+FN)     F1 = 2PR/(P+R)", 25, MUTED)
        self.play(FadeOut(mode), FadeOut(counts), Transform(score, formula.move_to(DOWN * 2.1)))
        self.say(
            "Micro suma conteos; macro promedia los F1 por tipo.\n"
            "BIO frente a Label Studio no explica por sí solo nuestro resultado débil.",
            hold=5,
        )
        self.play(
            FadeOut(note),
            FadeOut(ruler),
            FadeOut(ticks),
            FadeOut(gold),
            FadeOut(prediction),
            FadeOut(labels),
            FadeOut(overlap),
            FadeOut(score),
        )

    def llm_and_conclusion(self) -> None:
        self.topic(12, "El zero-shot del paper usa un LLM")
        self.say("El LLM genera menciones siguiendo instrucciones, sin una cabeza de tags BIO.")
        inputs = VGroup(
            text("códigos + descripciones", 25, TRAIN), text("nota + instrucción", 25, TRAIN)
        )
        inputs.arrange(DOWN, buff=0.4).move_to(LEFT * 4.1)
        llm = Circle(radius=0.9, color=GRADIENT, fill_opacity=0.1).move_to(LEFT * 0.1)
        llm_label = text("LLM local", 25, GRADIENT).move_to(llm)
        route = VGroup(*(Line(item.get_right(), llm.get_left(), color=TRAIN) for item in inputs))
        self.play(FadeIn(inputs), Create(llm), FadeIn(llm_label), Create(route))
        self.flow(route, TRAIN)
        output = text("{“text”: “dolor lumbar”,\n “label”: “HALLAZGO”}", 24, GOLD)
        output.move_to(RIGHT * 3.9)
        self.play(TransformFromCopy(llm, output))
        self.say(
            "Qwen3.5 del paper recibe instrucciones y devuelve citas JSON.\n"
            "Es zero-shot: sin demostraciones y sin ajustar sus pesos.",
            hold=4,
        )
        demos = text("+ ejemplos", 27, TEST).move_to(LEFT * 4 + DOWN * 1.5)
        self.play(FadeIn(demos))
        demo_route = VGroup(Line(demos.get_right(), llm.get_bottom(), color=TEST))
        self.flow(demo_route, TEST)
        self.say(
            "Nuestro fit LLM guarda ejemplos para el prompt: few-shot.\n"
            "Nuestro fit BERT sí aprende pesos de la cabeza. La interfaz no iguala los métodos.",
            hold=5,
        )
        self.play(
            FadeOut(inputs),
            FadeOut(route),
            FadeOut(llm),
            FadeOut(llm_label),
            FadeOut(output),
            FadeOut(demos),
        )
        answer = text("BIO: ya lo convertimos", 38, GOLD).move_to(UP * 1.2)
        heldout = text("600 train + 150 test, cinco modelos", 34, TEST).move_to(DOWN * 0.1)
        f1 = text("F1 de BETO: 750 predicciones test", 34, TRAIN).move_to(DOWN * 1.4)
        self.play(FadeIn(answer, shift=UP * 0.2))
        self.play(FadeIn(heldout, shift=UP * 0.2))
        self.play(FadeIn(f1, shift=UP * 0.2))
        self.say(
            "No tenemos sus IDs/folds exactos para una réplica idéntica.\n"
            "Compartir métricas no significa haber repetido el mismo experimento.",
            hold=6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    parser.add_argument(
        "--fps", type=int, choices=[15, 30, 60], help="Override the quality preset's frame rate."
    )
    parser.add_argument(
        "--pace", type=float, default=1, help="Scale all durations; 1 is the reading version."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/animations"))
    args = parser.parse_args()
    if args.pace <= 0:
        parser.error("--pace must be greater than zero")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    render_config: dict[str, object] = {
        "quality": f"{args.quality}_quality",
        "media_dir": str(output),
        "video_dir": str(output),
        "output_file": "BertSinMisterio",
        "disable_caching": True,
        "verbosity": "WARNING",
        "progress_bar": "none",
    }
    if args.fps is not None:
        render_config["frame_rate"] = args.fps
    with tempconfig(render_config):
        scene = BertSinMisterio(pace=args.pace)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    (output / "explicacion.md").write_text(READING_GUIDE, encoding="utf-8")
    (output / "capitulos.json").write_text(
        json.dumps(scene.chapter_times, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Video: {movie}\nGuide: {output / 'explicacion.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
