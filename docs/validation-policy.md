# Validación opcional con pocos ejemplos y revisión del usuario

Decisión de diseño del 20 de octubre de 2026. Los modos descritos son una
propuesta; todavía no se han añadido flags al CLI ni una API cliente/servidor.

## Qué queremos saber

Los ejemplos sirven para enseñar el comportamiento deseado y para comprobar
que los cambios conservan casos ya resueltos. Una evaluación separada responde
otra pregunta: qué ocurre en notas que el sintetizador no ha visto como ejemplos.
Ambas necesidades son válidas; no hacen falta dos cores.

Las reglas también pueden sobreajustar. Una lista de frases vistas puede
acertar todos los ejemplos y omitir variantes nuevas. Una condición demasiado
amplia puede funcionar en train y añadir falsos positivos fuera de él. Aprender
abreviaturas o un vocabulario cerrado es útil; su utilidad no convierte el F1
de entrenamiento en una medida de generalización. La cantidad de etiquetas
emitidas tampoco sustituye a precisión y recall sobre las etiquetas correctas.

Con veinte ejemplos puede ser razonable usarlos todos para aprender. Apartar
cuatro reduce la supervisión y puede dejar categorías raras sin ejemplos; esos
cuatro también pueden dar una estimación muy variable. No tenemos evidencia
propia de que 16/4 sea mejor que 20/0 en nuestra tarea.

## Un flujo simple primero

1. El usuario aporta esquema, instrucciones y ejemplos autorizados para fit.
2. El servidor sintetiza y puede refinar internamente sobre esos ejemplos.
3. Devuelve un artefacto candidato al terminar el job.
4. El cliente carga, predice y puntúa sobre los ejemplos elegidos. Muestra si son
   casos conocidos o validación separada, y permite inspección manual.
5. El usuario acepta o envía feedback/correcciones con la versión base para otro
   job. Se conservan la versión anterior y los casos de regresión previos.

El cliente usa las mismas operaciones `load`, `predict` y `score`. Calcular
métricas no obliga a volver a ejecutar fit. La CLI ya permite cargar y predecir
sin fit para los métodos registrados; falta integrar RuleChef en ese registro
y exponer revisión de artefactos.

La operación de revisión necesita el estado base y las correcciones, no solo
«vuelve a aprender con este error». Si el servidor no conserva artefactos,
el cliente envía el payload base junto a la petición. Los ejemplos previos
siguen funcionando como regresiones; los comentarios en lenguaje natural
complementan las anotaciones corregidas.

## Tres políticas, un contrato

| Política propuesta | Comportamiento |
|---|---|
| `validation=none` | Aprender de todos los ejemplos; reportar ajuste/regresión y revisión humana. |
| `validation=split` | El cliente aparta ejemplos antes de fit y conserva su gold localmente. |
| `validation=provided` | Aprender de todos los ejemplos de train y evaluar con otro conjunto aportado. |

No fijar un 20% obligatorio. Guardar IDs, semilla/grupos cuando se usen,
soporte por etiqueta y si cada caso se mostró al sintetizador. Los documentos
del mismo paciente deben permanecer juntos cuando se disponga de ese vínculo.

Una automatización posterior puede enviar artefactos candidatos, recibir
métricas locales y elegir otra revisión. No es necesaria para la primera PoC.
El lugar donde se calculan las métricas y el uso estadístico de los datos son
decisiones distintas: un dev local también es adaptativo si sus resultados
orientan cambios en el fitter. La privacidad de sus textos no garantiza que
su puntuación siga siendo una estimación independiente.

Si una corrección se envía como ejemplo, deja de ser heldout para ese candidato.
Para comparaciones de dev, retirar ese caso, versionar el split y reevaluar
ambas versiones sobre los mismos ejemplos restantes. Mantener un test final
ajeno a las iteraciones cuando se necesite una medida final independiente.

## Evidencia disponible y sus límites

La ablación de RuleChef en TAB reporta 63,2/38,2 F1 para tipos de formato/
semánticos al refinar sobre train y 83,5/44,4 al añadir aceptación por holdout.
Usa 1.000 chunks de entrenamiento y 600 de test. Apoya investigar validación,
pero no determina el mejor split para veinte notas clínicas ni elimina la
variabilidad entre síntesis. [Paper, §5.2](https://arxiv.org/html/2607.01293v1).

Optimizar repetidamente un criterio de selección estimado con pocos datos
puede sobreajustar también ese criterio, no solo el entrenamiento. Ese es el
problema estudiado por
[Cawley y Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html).
La adaptación repetida al holdout también puede sobreajustarlo, como analiza
[Dwork et al., 2015](https://arxiv.org/abs/1506.02629).
Estos resultados justifican distinguir dev y evaluación final; no obligan a
implementar validación distribuida ni algoritmos especiales para la PoC.

Nuestra evidencia directa: small obtiene 93,95% strict micro en las diez notas
de train y 87,21% en las 789 restantes; large, 93,30% y 87,10%. Hay diferencias
de cobertura y distribución de etiquetas. Es una brecha de generalización,
no una prueba de que reservar cuatro ejemplos mejore el resultado. La auditoría
está en [rulechef-paper-audit.md](methods/rulechef-paper-audit.md).

## Estado del experimento

La matriz de cinco rondas implementada compara las políticas siguientes y
añade síntesis inicial y coordinación de tres/ocho rondas. Se lanzó el 20 de
octubre de 2026. A 21 de octubre hay cuatro resultados de la primera ronda,
de Sol, incluidos holdout y coordinación de hasta tres iteraciones. La variante de ocho
se detuvo por el límite de uso de Codex; faltan esa variante y las rondas 2–5.
El refinamiento con holdout produjo dos parches sin mejorar dev, por lo que
el artefacto final conservó el mejor estado inicial. Las métricas están en
[resultados agregados](results/rulechef-ablation.md); el protocolo, los límites
y los comandos, en [rulechef-experiments.md](methods/rulechef-experiments.md).

La comparación con `gpt-6-luna / low` usa exactamente los mismos repartos y
código, y conserva sus resultados y artefactos en otra matriz. La tabla pública
incluye solo resultados agregados completos, con la fecha del corte; el estado
de ejecución se consulta localmente en el directorio privado de cada matriz.

Comparar dos políticas con un presupuesto de veinte notas anotadas:

- A: veinte para síntesis y refinamiento sobre train.
- B: dieciséis para síntesis y cuatro para seleccionar candidatos; dev excluido
  también de la petición de síntesis inicial.

Mantener esquema, instrucciones y formatos, registrar las llamadas reales y puntuar
ambas políticas sobre un test externo a esas veinte notas y a las correcciones.
Repetir con varias selecciones de ejemplos y varias síntesis, emparejando las
cohortes. Medir F1 micro/macro, precisión, recall, resultados por etiqueta,
coste y tiempo. Eso responde si, con nuestro presupuesto, compensa el menor
train frente a una selección por dev. Un control adicional de dieciséis notas
sin dev permitiría separar el efecto del tamaño de train del de selección.

El experimento de síntesis necesita excluir del prompt la implementación de las
reglas humanas y los errores de su test. Las notas de la cohorte actual ya son
exploratorias; no presentarla como una confirmación independiente tras usarla
para diseñar mejoras.

También se puede evaluar un procedimiento mediante cross-validation y después
ajustar un artefacto final con todos los ejemplos. Las métricas de los folds
describen ese procedimiento; no son una evaluación independiente del artefacto
final entrenado con todos.
