# RuleChef: prompts, aprendizaje y cinco rondas

Implementación del 20 de octubre de 2026. Los resultados anteriores de diez
notas siguen documentados en [rulechef.md](rulechef.md); no son la ablación nueva.

## Qué aporta RuleChef y qué añadimos

Sí usamos los prompts de la dependencia `rulechef==0.2.2`. No hemos copiado el
apéndice al adaptador ni sustituido su sintetizador por un agente propio.
El [apéndice A](https://arxiv.org/html/2607.01293v1#A1) describe sus plantillas;
el [código upstream](https://github.com/KRLabsOrg/rulechef) permite inspeccionarlas.

| Pieza | Quién la construye/ejecuta |
|---|---|
| Cabecera con tarea, esquema y etiquetas | `RuleLearner` / `PromptBuilder` de RuleChef |
| Ejemplos, evidencia, formatos y JSON solicitado | RuleChef |
| Síntesis conjunta o una petición por clase | RuleChef |
| Selección de errores, variantes del prompt de parche y mezcla por nombre | RuleChef |
| Prompts de coordinador, crítico y auditor | `AgenticCoordinator` de RuleChef |
| Ejecución regex / spaCy y resolución de conflictos | `RuleExecutor` de RuleChef |
| Glosas clínicas e instrucciones iniciales | `TaskSpec.instructions` y configuración del experimento |
| Explicar FUZZY y el runtime español disponible | Nuestro adaptador |
| Transporte Codex o cliente compatible con OpenAI | `methods/rulechef_client.py` |
| Aceptación con dev explícito y offsets exactos | Nuestro adaptador, reutilizando las propuestas nativas |
| Splits, aislamiento de test y métricas finales | Runner experimental y evaluador del proyecto |

La dependencia tiene licencia Apache-2.0. La librería aporta un sistema
reutilizable de propuestas, ejecución y aprendizaje, no un optimizador que
enumere automáticamente todas las combinaciones de regex y FUZZY.

Nuestro transporte añade solamente una instrucción de actuar como modelo de
completado y devolver el JSON pedido por los mensajes. Antes forzaba un esquema
de respuesta con `rules` incluso para llamadas de coordinación: eso se eliminó.
El coordinador puede devolver `guidance/should_continue`; el auditor, `actions`;
el crítico, feedback. Una prueba real aislada confirmó este transporte con
`gpt-6.1-sol`, sin herramientas.

Los prompts completos de cada llamada se guardan en `request.json` y, para
Codex, también en `prompt.txt`. Las respuestas originales y los eventos permiten
comprobar qué propuso el modelo y qué aceptó el parser. Estas trazas son datos
de investigación privados; no forman parte de un artefacto de cliente público.

## Formatos y spaCy

Se permiten regex y patrones spaCy conjuntamente. La descripción explica
`FUZZY/FUZZY1`, sus riesgos con cadenas cortas y el uso de regex para límites
dentro de un token. El LLM decide qué proponer; permitir un formato no obliga
a generar reglas de ese formato ni constituye una comparación de alternativas.

`blank:es` crea `spacy.blank("es")`: tokenizador y vocabulario, sin pesos
entrenados. Sirve para patrones lexicales y FUZZY. No produce lemas, POS ni
dependencias. `es_core_news_sm` y `es_core_news_lg` sí producen esas anotaciones.
`use_spacy_ner=False` solo desactiva entidades genéricas/`ENT_TYPE`; no desactiva
Matcher ni FUZZY. Véase la [documentación de spaCy](https://spacy.io/usage/rule-based-matching#fuzzy).

Los artefactos nuevos que solo contienen regex no requieren cargar un modelo
spaCy al inferir, aunque se permitiera durante el fit. Los artefactos antiguos
siguen siendo legibles. La cadena `model_version` nueva es `rulechef`; la
versión real de la dependencia y del generador están en provenance.

## Reparaciones, aceptación y coordinación

Una reparación propone un parche de reglas a partir de fallos en train.
RuleChef puede añadir, reemplazar por nombre y borrar reglas. Se ejecuta el
candidato y se acepta si F1 cae como máximo 0,005 o mejora la precisión,
siguiendo la política upstream. Se conserva por separado el mejor snapshot
por F1; una reparación aceptada temporalmente no tiene por qué ser el artefacto
final. Tres u ocho iteraciones son máximos; puede parar antes.

Con dev explícito, el sintetizador y el crítico reciben solo train. Dev decide
aceptación, mejor snapshot y estadísticas de desempate. Esta aceptación adapta
el sistema a dev: sus métricas no sustituyen al test externo. El coordinador
propone prioridades, el crítico escribe feedback cada dos rondas y el auditor
propone consolidación cada tres y al final. Sus cambios también pasan el gate
de dev; al final hay poda leave-one-out sin llamadas al modelo.

El F1 global de dev no se envía al LLM, tampoco como un número aislado. El
prompt de parche recibe fallos concretos, resumen de modos de error, métricas
por etiqueta y reglas actuales de train. En coordinación agentic, las métricas
que recibe el coordinador también son de train. Dev orienta la búsqueda al
seleccionar el estado desde el que continúa, conservar el mejor y decidir
paradas; no aporta ejemplos para explicar cómo reparar los errores.

Hay un detalle del artículo que todavía no se replica: su §3.2 y figura 2
muestran precisión por regla medida en dev dentro del prompt de parche.
`_rule_prompt_entry` de la dependencia 0.2.2 no incluye esas estadísticas.
Nuestro bucle las mide y guarda al terminar para resolver conflictos durante
inferencia, sin incorporarlas entre parches. Por tanto, reproducimos aceptación
por holdout, pero no toda la información de dev descrita en ese prompt del
[paper](https://arxiv.org/html/2607.01293v1#S3.SS2).

La pequeña orquestación en `methods/rulechef_learning.py` es necesaria porque
la API 0.2.2 no recibe un dev externo: sintetiza antes de dividir y puede
cancelar silenciosamente un holdout pequeño. No se modifica la dependencia.
La evaluación del conjunto y el stamping se corrigen a offsets exactos;
los helpers nativos de crítico/auditor conservan sus estadísticas individuales
basadas en texto para orientar propuestas. La decisión final usa offsets.

Una ejecución real de la síntesis de 16 notas detectó otro caso del runtime
0.2.2: una captura dentro de un lookahead puede contener texto aunque el match
completo tenga longitud cero. El adaptador obtiene sus posiciones de
`Match.span(grupo)`; no busca la cita en la nota ni modifica las reglas generadas.
El arreglo tiene regresiones de Unicode, repeticiones, lookahead/lookbehind y
recarga. Una recuperación debe congelar una revisión nueva y conservar las
trazas originales; las respuestas previas solo pueden reutilizarse si coinciden
exactamente sus peticiones y pasan la auditoría de aislamiento.

`fit` y `refine` siguen usando el mismo core. `refine` parte de una copia del
artefacto inicial para no sobrescribir el incumbente. Todavía falta un contrato
general de correcciones/feedback del usuario y su integración en la CLI pública.

## Diseño de la ablación

| Configuración | Ejemplos de síntesis/parches | Aceptación | Máximo de rondas |
|---|---:|---|---:|
| `oneshot` | 20 | Sin parches | 0 |
| `train_refine` | 20 | Train | 3 |
| `holdout` | 16 | 4 dev | 3 |
| `agentic3` | 16 | 4 dev + coordinación | 3 |
| `agentic8` | 16 | 4 dev + coordinación | 8 |

En cada ronda se eligen 20 notas de un total de 799 y las otras 779 son test
común para las cinco configuraciones. Por tanto, veinte es el presupuesto
etiquetado total, incluido dev cuando se usa. Las dos primeras filas usan
todas las veinte. Las tres restantes usan 16/4, dividido antes de sintetizar.
No enviamos test al modelo ni sus errores a los parches.

Se reutiliza una síntesis inicial para las filas que usan las mismas notas:
una de 20 y otra de 16 por ronda. Esto reduce coste y permite comparar las
variantes desde las mismas reglas. Cada tabla cuenta las llamadas iniciales
como parte de su configuración; no se deben sumar filas para calcular llamadas
únicas ejecutadas. La síntesis por clase hace 42 llamadas si se observan las
42 etiquetas; «one-shot» significa una pasada, no una sola petición.

La selección automática prioriza cobertura de etiquetas raras y luego déficits
respecto a frecuencias documentales. No es una muestra uniforme ni garantiza
proporciones perfectas. Se mantienen juntos los duplicados por texto normalizado.
No hay identificadores de paciente para garantizar separación por paciente.
El manifiesto registra IDs, hashes, frecuencias de documentos/menciones y
etiquetas ausentes en cada partición. En los cinco repartos preparados, tanto
las 20 notas como las 16 de síntesis cubren las 42 etiquetas; dev cubre 38–40.

Son cinco repeticiones few-shot, con solapamiento entre rondas. La desviación
estándar es descriptiva, no un intervalo de confianza de cinco folds
independientes. `--split-mode fivefold` permite pasar a particiones externas
disjuntas cercanas a 80/20; el split interno de dev sigue siendo configurable.
Todo el corpus ya se ha usado exploratoriamente: no es un test nuevo intacto.

Se reproduce la estructura de la ablación, no sus números ni todas sus
condiciones. Nuestro generador es distinto y usamos español, 42 etiquetas,
20 notas, regex+spaCy y evaluación exacta. El benchmark upstream permite
regex+code aunque el texto del artículo dice centrarse en regex. No ejecutamos
Python arbitrario. Fmt/Sem son grupos de TAB; nuestra tabla informa strict
micro/macro, overlap micro, reglas y llamadas, además del detalle por etiqueta.

## Ejecutar y reanudar

Con el Codex CLI autenticado y las dependencias instaladas:

```bash
uv sync --locked --extra rulechef --extra spacy
.venv/bin/python experiments/rulechef_ablation.py \
  --output runs/rulechef_ablation20_new --stage prepare \
  --model gpt-6.1-sol
nohup .venv/bin/python runs/rulechef_ablation20_new/source/experiments/rulechef_ablation.py \
  --output "$PWD/runs/rulechef_ablation20_new" --stage run \
  > runs/rulechef_ablation20_new/runner.log 2>&1 < /dev/null &
```

`prepare` congela código, configuración e inputs. `run` ejecuta el snapshot en
procesos separados para fit y test, actualiza `status.json` y `report.md` después
de cada celda y salta celdas terminadas al reanudarse. Un fallo detiene la matriz;
se conserva el log. Un inicial completado se reutiliza, pero un fit inicial
interrumpido a mitad de sus 42 llamadas todavía se repite: no hay checkpoint
por etiqueta. Lo mismo ocurre con un refinamiento interrumpido dentro de celda.
No lanzar dos runners contra el mismo directorio simultáneamente.

Para menos llamadas se puede preparar otro experimento con
`--synthesis-strategy bulk`; no mezclar sus resultados con los de `per_class`.
La matriz inicial usa `gpt-6.1-sol` con razonamiento `high`; los pilotos
anteriores usaban `xhigh`. El 21 de octubre se lanzó una segunda matriz con
`gpt-6-luna` y `low`, conservando exactamente splits, código y prompts. Esta
comparación cambia modelo y esfuerzo conjuntamente. Para preparar esa variante
en otro directorio vacío:

```bash
.venv/bin/python experiments/rulechef_ablation.py \
  --output runs/rulechef_ablation20_luna_low_new --stage prepare \
  --model gpt-6-luna --reasoning-effort low
```

El modelo más barato reduce el precio por token; no reduce por sí solo el
tamaño del prompt ni las 42 llamadas de una síntesis por clase. El número real
de tokens de salida y razonamiento se mide en cada ejecución.

Para un servidor MLX/LM Studio que ofrezca Chat Completions, el mismo runner
acepta `--backend openai --endpoint http://127.0.0.1:8080/v1 --model NOMBRE`.
Si el servidor no autentica, `OPENAI_API_KEY=local` satisface el SDK. No necesita
OpenCode ni Codex ni bubblewrap: es una petición de texto, no un agente con
herramientas. Hay que comprobar compatibilidad de JSON y límites de contexto
del servidor concreto antes del fit completo; no se ha validado este recorrido
nuevo en Mac Studio.

Cada resultado conserva métricas por etiqueta, predicciones, fallos, reglas,
llamadas, tokens disponibles, tiempo de modelo, carga e inferencia y RSS de
procesos. RSS no mide memoria del servidor remoto ni suma el árbol de procesos.
Los campos de caché y razonamiento pueden ser subconjuntos de input/output;
no se suman todas las claves de usage como si fueran tokens distintos.
El coste monetario queda nulo: no se convierte uso de suscripción en una tarifa
API supuesta. Las llamadas Codex conservan el preflight bubblewrap y los eventos
para verificar que no tuvieron herramientas ni acceso al repositorio.

## Resultados publicables y privacidad

La [tabla de resultados agregados](../results/rulechef-ablation.md) reúne las
configuraciones completadas, las pendientes y los recursos medidos. Es la
referencia pública para esta ablación; no se copian los JSON privados del runner.

El documento publica solo métricas globales, cantidades, configuración general
y recursos. Notas, fragmentos, IDs, hashes privados, splits por documento,
prompts, respuestas, errores individuales y reglas/lexicones aprendidos se
mantienen locales. Un artefacto derivado de datos privados también se trata como
privado: no basta con omitir las notas originales para hacerlo publicable.

## Banking77 y próximos baselines

La tabla 6 muestra precisión alta pero cobertura/recall parciales para las
reglas. Eso explica su menor F1 frente al LLM sin demostrar overfit del dataset.
El artículo además dice 25 clases en §4.2 y 5 en §5.5: la configuración exacta
necesita aclaración antes de una reproducción de esa tabla.

La [ficha de DeBERTa](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33)
incluye Banking77 en su entrenamiento y advierte que sus métricas sobre esos
datasets no son estrictamente zero-shot. No demuestra solapamiento con las
consultas concretas del paper, pero sí exposición previa al dominio/tarea.
Además es un clasificador inglés de texto completo, no un extractor de spans.
Un baseline NLI nuestro necesitaría proponer candidatos y evaluar sus límites,
además de elegir un modelo adecuado para español.
