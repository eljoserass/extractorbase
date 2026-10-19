# Piloto de reglas spaCy generado por un agente

`methods/hard_spacy.py` implementa `HardSpacyMethod` con la interfaz habitual
`fit`, `dump`, `load` y `predict`. Un agente Codex independiente escribió el
método y realizó una sola pasada de corrección después de su primera evaluación.
La generación se hizo con `gpt-6.1-sol`, no con el modelo local de los experimentos
anteriores.

El agente recibió exclusivamente las notas 1–10 y sus anotaciones seleccionadas
con la política `clinician` (revisores 3/4), el esquema de Label Studio de 42
etiquetas, sus definiciones y las interfaces genéricas de datos y evaluación.
Son 216 entidades anotadas, con soporte para 31 de las 42 etiquetas.

La ejecución usó un filesystem de `bubblewrap` con una lista explícita de
montajes: workspace preparado, intérprete y dependencias, binario del agente y
estado de autenticación temporal. El repositorio original, los montajes del host,
las demás notas, reglas previas, predicciones y resultados anteriores quedaron
fuera. Los archivos de entrada y el evaluador estaban montados en solo lectura.
El proceso del harness mantuvo la conexión necesaria al modelo; sus comandos
tenían la red desactivada por la política de Codex, sin búsqueda web ni MCPs.

No hubo conjunto de desarrollo ni evaluación sobre otras notas. El feedback
durante la generación procede de las mismas diez notas de entrenamiento.

## Resultado sobre entrenamiento

| Métrica estricta por entidad | Resultado |
| --- | ---: |
| Precisión micro | 98,11% |
| Recall micro | 96,30% |
| F1 micro | 97,20% |
| Entidades gold | 216 |
| Aciertos exactos | 208 |
| Falsos negativos | 8 |
| Falsos positivos | 4 |

Estas cifras son de ajuste al entrenamiento y no estiman generalización.
Se conservaron las anotaciones originales, incluidas sus inconsistencias de
etiquetado y límites. El informe del agente detalla los errores restantes.
Las relaciones se validan estructuralmente, pero el evaluador no las puntúa.

## Funcionamiento y límites

Usa `spacy.blank("es")`, un tokenizador que conserva offsets y `Matcher`.
`fit` deriva diccionarios de menciones de las anotaciones recibidas. Las reglas
añaden valores numéricos, fracciones y polaridad próximos a pruebas o índices,
resuelven determinados solapamientos y emiten relaciones `valor` cuando hay
soporte en entrenamiento. No requiere pesos preentrenados.

El artefacto JSON contiene el esquema y el estado aprendido; no contiene notas
completas, IDs de documentos ni tablas de offsets gold. Las etiquetas sin soporte
no se emiten. No hay atribución paciente/familia ni interpretación temporal o
negación clínica general. Las decisiones de mayoría también pueden propagar
errores de anotación, como la clasificación ambigua de `AR` en estas diez notas.

Se comprobó la carga en otro proceso con el directorio de entrenamiento y el
estado del agente ocultos y con la red aislada. Produce exactamente las mismas
predicciones a partir del artefacto y las entradas de texto.

## Uso

Instalar la dependencia opcional conservando los demás paquetes del entorno:

```bash
uv sync --inexact --extra spacy
```

El método se utiliza directamente mediante la interfaz `Method`:

```python
from pathlib import Path
from methods.hard_spacy import HardSpacyMethod

method = HardSpacyMethod()
artifact = method.fit(train_examples, task_spec)
method.dump(artifact, Path("my_artifact"))

artifact = HardSpacyMethod().load(Path("my_artifact"))
predictions = method.predict(artifact, document_inputs)
```

El artefacto del piloto está en
`runs/hard_spacy_train10_2026-10-19/artifact/`. Esa carpeta contiene también:

- `PROMPT.md`, `AGENTS.md` y `provenance.json`: tarea exacta, entradas, modelo,
  comandos y hashes.
- `isolation_audit.json` y `fresh_process_reload.json`: comprobaciones de acceso
  y carga sin entrenamiento ni agente.
- `training_predictions.json`, `training_metrics.json` y `training_errors.json`:
  resultados completos sobre las diez notas.
- `implementation_notes.md` y `agent_tests/`: explicación y las seis pruebas
  realizadas por el agente.
- `source/hard_spacy.py` y `agent_hard_spacy.py`: versión integrada congelada y
  código original entregado por el agente, respectivamente.

Las notas y los resultados se mantienen en `runs/`, excluido de Git. El workspace
y el estado temporal del agente se eliminan después de guardar el experimento.
