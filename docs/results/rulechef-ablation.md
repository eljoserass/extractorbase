# RuleChef: ablación en un corpus privado

Resultados parciales revisados el 2026-10-21 a las 07:44 UTC.
Se adaptó la estructura de la tabla de ablación de RuleChef a una tarea
clínica en español. Las cinco repeticiones previstas aún no han terminado.

## Protocolo

- Corpus privado de 799 notas, con 42 etiquetas de entidades.
- Cada ronda: 20 notas de aprendizaje y 779 notas comunes de test.
- Síntesis inicial y refinamiento sobre train: las 20 notas para aprender.
- Aceptación en dev y variantes agentic: 16 train y 4 dev, separados antes de sintetizar.
- El LLM recibe ejemplos y errores de train; dev selecciona candidatos y el test final puntúa el artefacto.
- Síntesis por clase, regex y spaCy permitidos, con tokenizer español sin pesos.
- Generadores: `gpt-6.1-sol / high` y `gpt-6-luna / low`; dependencia `rulechef==0.2.2`.
- F1 por entidad con posiciones exactas; overlap exige coincidencia de etiqueta y solapamiento.

Las dos matrices conservan exactamente splits, código y prompts. Cambian
conjuntamente modelo y esfuerzo de razonamiento. No es una comparación que
aísle únicamente el modelo.

La selección automática prioriza cobertura de etiquetas raras y después
déficits respecto a sus frecuencias documentales; no garantiza proporciones
idénticas. En los cinco repartos preparados, tanto las 20 notas de aprendizaje
como las 16 de síntesis cubren las 42 etiquetas. Dev cubre entre 38 y 40.
Los duplicados por texto normalizado se mantienen juntos.

Son cinco repeticiones few-shot con solapamiento entre rondas, no cinco
particiones externas disjuntas. La cohorte es exploratoria y no se dispone
de agrupación por paciente. El [protocolo detallado](../methods/rulechef-experiments.md)
describe la separación, los prompts nativos y las diferencias con upstream.

## Primera ronda: resultados de test

Cada fila terminada se evalúa sobre las mismas 779 notas de la primera ronda.
Una sola ronda no permite medir estabilidad entre selecciones de ejemplos.

### Sol / high

| Configuración | Strict micro F1 | Strict macro F1 | Overlap micro F1 | Reglas | Llamadas LLM |
|---|---:|---:|---:|---:|---:|
| Síntesis inicial | 91,25% | 90,03% | 92,45% | 98 | 42 |
| + Refinamiento en train | 91,51% | 90,27% | 92,68% | 99 | 43 |
| + Aceptación en dev | 89,39% | 87,60% | 90,83% | 91 | 44 |
| + Coordinación, máximo 3 iteraciones | 82,34% | 66,65% | 82,91% | 49 | 51 |
| Coordinación, máximo 8 iteraciones | pendiente | pendiente | pendiente | — | — |

Matriz detenida tras un fallo: 4 de las 25 celdas completas en el corte indicado.

### Luna / low

| Configuración | Strict micro F1 | Strict macro F1 | Overlap micro F1 | Reglas | Llamadas LLM |
|---|---:|---:|---:|---:|---:|
| Síntesis inicial | 83,51% | 83,70% | 84,82% | 75 | 42 |
| + Refinamiento en train | 83,80% | 82,53% | 85,51% | 75 | 45 |
| + Aceptación en dev | 84,12% | 79,90% | 85,87% | 71 | 45 |
| + Coordinación, máximo 3 iteraciones | 78,67% | 61,08% | 79,75% | 38 | 51 |
| Coordinación, máximo 8 iteraciones | pendiente | pendiente | pendiente | — | — |

Matriz en ejecución: 4 de las 25 celdas completas en el corte indicado.

Las llamadas incluyen la síntesis inicial compartida. No se deben sumar
filas para estimar llamadas únicas. Una pasada por clase requiere 42 llamadas
iniciales. Los máximos de iteraciones permiten paradas anticipadas.

## Lectura de los resultados

En las configuraciones inicial y con refinamiento sobre train de la primera
ronda, Luna / low reduce el tiempo de llamadas al modelo y los tokens de
salida, con un F1 inferior a Sol / high. El volumen de entrada es parecido:
cambiar de modelo no acorta por sí solo los prompts.

En esta ronda, Luna mejora ligeramente strict micro al usar holdout, pero
baja strict macro. Con Sol, reservar cuatro notas no mejora el test. Cambian
tanto el tamaño del train como las reglas iniciales; no permite atribuir la
diferencia únicamente a aceptación ni concluir que dev sea inútil. Las
variantes agentic y las demás rondas deben juzgarse por sus resultados reales.

En Sol, el refinamiento con aceptación en dev produjo dos parches con el
mismo strict micro F1 de validación, 90,98%. Se aceptaron como estados de
trabajo, pero no mejoraron el mejor F1: el artefacto final conservó las 91
reglas iniciales sobre 16 notas. Dev no se envió como ejemplos al sintetizador.

Se replica la estructura de configuraciones de la [tabla del paper](https://arxiv.org/html/2607.01293v1#S5.SS2),
con adaptaciones explícitas. No se reproducen sus condiciones ni sus valores:
cambian corpus, idioma, generador, presupuesto de ejemplos, formatos y etiquetas.
Fmt/Sem de TAB no equivale a nuestros promedios clínicos micro/macro. Falta
el feedback de precisión por regla en dev dentro del prompt descrito en el
artículo; véase la [diferencia con la librería 0.2.2](../methods/rulechef-experiments.md#reparaciones-aceptación-y-coordinación).

## Recursos agregados de la primera ronda

| Generador | Configuración | Tokens entrada | Tokens salida | Tiempo de modelo | Inferencia de test | RSS del proceso de test |
|---|---|---:|---:|---:|---:|---:|
| Sol / high | Síntesis inicial | 1,943,085 | 31,742 | 15.3 min | 0.83 s | 160.0 MiB |
| Sol / high | + Refinamiento en train | 2,008,422 | 37,153 | 17.6 min | 0.90 s | 159.9 MiB |
| Sol / high | + Aceptación en dev | 1,797,803 | 38,250 | 18.9 min | 1.43 s | 159.5 MiB |
| Sol / high | + Coordinación, máximo 3 iteraciones | 2,007,746 | 63,139 | 29.6 min | 0.96 s | 157.4 MiB |
| Luna / low | Síntesis inicial | 1,911,404 | 15,296 | 5.3 min | 1.06 s | 159.4 MiB |
| Luna / low | + Refinamiento en train | 2,219,957 | 19,480 | 6.0 min | 0.69 s | 158.4 MiB |
| Luna / low | + Aceptación en dev | 1,900,684 | 21,215 | 6.1 min | 0.72 s | 159.2 MiB |
| Luna / low | + Coordinación, máximo 3 iteraciones | 2,027,730 | 25,368 | 7.2 min | 0.40 s | 157.3 MiB |

Los tiempos de modelo suman las llamadas incluidas en cada configuración,
también cuando se reutilizaron respuestas de síntesis. La inferencia es local
en CPU, sin llamadas LLM; las mediciones excluyen arranque/importaciones,
carga del dataset y puntuación. RSS no mide memoria del modelo remoto ni la
suma de un árbol de procesos. Caché y razonamiento son subconjuntos de los
tokens de entrada/salida. No se estima coste monetario desde la suscripción.

## Estado y reproducción

La matriz Sol se detuvo por el límite de uso de Codex en la primera llamada
de refinamiento de la variante de ocho iteraciones. La matriz Luna usa su
propio directorio y sus propios artefactos; no reutiliza reglas de Sol.
El modo clásico de cinco folds externos cercanos a 80/20 está implementado
mediante `--split-mode fivefold`, pero no se ha ejecutado.

Las variantes completadas no tuvieron fallos de inferencia. La síntesis de
Sol para dev requirió una corrección del runtime para capturas regex en
lookarounds de longitud cero. Se reutilizaron 42 respuestas auditadas tras
comprobar igualdad exacta de sus peticiones, sin regenerar reglas ni usar
errores de test. Las dos evaluaciones previas conservaron sus predicciones.

El código y la configuración general permiten repetir el procedimiento con
datos autorizados. Las entradas privadas no se distribuyen; el repositorio
público no permite reconstruir exactamente esta cohorte.

## Privacidad de lo publicado

Este documento contiene solo métricas globales, cantidades agregadas,
configuración general y recursos. Notas, citas, anotaciones, IDs, hashes
privados, splits por documento, prompts, respuestas, errores individuales,
reglas aprendidas, lexicones y artefactos se mantienen fuera de Git.
Las reglas aprendidas pueden memorizar vocabulario de las entradas y se
tratan como información privada aunque se ejecuten localmente.
