# Auditoría agregada de reglas: alcance y conclusiones

Resumen de la inspección del 20 de octubre de 2026. Las fuentes suministradas,
las reglas aprendidas y la auditoría detallada permanecen en almacenamiento
privado. Este documento publica conclusiones técnicas y métricas globales;
no incluye citas de notas, identificadores, hashes privados, patrones aprendidos
ni vocabularios derivados del corpus.

## Comparación histórica sobre una cohorte común

Los pilotos siguientes se evaluaron sobre las mismas 789 notas, después de
aprender con diez ejemplos. Son resultados exploratorios de una generación
por configuración, no una validación cruzada.

| Sistema | Strict micro F1 | Strict macro F1 |
|---|---:|---:|
| Reglas suministradas, reproducción local | 94,15% | 94,94% |
| RuleChef, regex + spaCy small permitido | 87,21% | 80,68% |
| RuleChef, regex + spaCy large permitido | 87,10% | 81,81% |

La muestra exacta del manuscrito no se ha recuperado. El gold disponible procede
de correcciones de preetiquetados: estas cifras miden acuerdo con esa revisión,
y no permiten afirmar una reproducción exacta del resultado del manuscrito.
El histórico completo de métricas globales está en [rulechef-pilots.md](rulechef-pilots.md).

## Por qué small y large no aislaron el efecto del modelo

Las dos síntesis permitían regex y spaCy, pero produjeron solo regex: 47 y 56
reglas respectivamente. Los pipelines entrenados procesaron cero documentos.
Ejecutar cada artefacto sin cargar esos pesos produjo predicciones idénticas
sobre toda la cohorte. Las diferencias de F1 comparan reglas generadas distintas;
no demuestran una contribución de las anotaciones lingüísticas de small/large.

## Cobertura y tipos de error

Los diez ejemplos cubrían 31 de las 42 etiquetas. Las categorías sin ejemplos,
variantes léxicas, erratas y convenciones de límites explican parte de la brecha.
El análisis detallado incluye decisiones por nota, por lo que se mantiene privado.

Las reglas suministradas consultaban atributos léxicos de los tokens y podían
funcionar sin lemas, POS ni dependencias aprendidas. El script original sí
cargaba un pipeline large; la reproducción local usó un tokenizer sin pesos,
conservando la tokenización y lógica de posprocesado inspeccionadas. No se han
publicado las reglas originales ni las generadas.

Las diferencias observadas afectan a cobertura de variantes, tolerancia de
erratas, tokenización, selección de spans y asociación contextual de valores.
El original también produce falsos positivos: no es una especificación clínica
infalible que deba copiarse. No se modificaron los artefactos de estos pilotos
usando errores del test.

## Qué cambia en los experimentos nuevos

Las últimas síntesis mixtas del piloto no tenían guía sobre FUZZY ni rondas de
reparación. Permitir spaCy no obligaba al LLM a comparar alternativas. No conocemos
la razón interna por la que eligió regex.

`spacy.blank("es")` proporciona tokenizer y vocabulario sin componentes
entrenados. Los patrones lexicales y FUZZY pueden funcionar con ese runtime;
una tolerancia mayor también puede aumentar falsos positivos.
Véanse la [documentación de idiomas](https://spacy.io/usage/models#languages)
y [FUZZY](https://spacy.io/usage/rule-based-matching#fuzzy).

El adaptador actual admite dev explícito antes de sintetizar, guía de formatos,
parches y coordinación. La ablación nueva usa veinte ejemplos y conserva el test
externo al aprendizaje. Su [protocolo](rulechef-experiments.md) y
[resultados agregados](../results/rulechef-ablation.md) distinguen lo ejecutado
de lo pendiente. La política de validación y el flujo de revisión del usuario
se describen en el [roadmap](../architecture-roadmap.md).
