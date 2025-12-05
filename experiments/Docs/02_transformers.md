# Experimento 3: ir a LSA-T con modelos más grandes

Ya es claro que necesito un modelo gloss-free o semi-gloss-free.

## Idea: usar LSA-T + segmentación débil
Para tener un baseline, mi idea es:
- Usar LSA-T (videos continuos con subtítulos en español).
- Convertir subtítulos → glosas usando NLP.
- Usar esas glosas para **segmentación débil**:
  - no hay alineación perfecta,
  - pero sí una secuencia aproximada de glosas.

Esto permite entrenar un modelo que:

- entiende estructura temporal,
- no depende de ventanas artificiales,
- no exige anotación exacta.

### Cambios sobre la arquitectura original de LSA-T

El paper original usa:

- 3 convoluciones temporales con kernel tamaño 1 → **solo 3 features por frame**  

Siento que es demasiada reducción, para eso quiero usar **~50 features** temporales en vez de 3, o pasar los landmarks directamente al modelo secuencial sin convoluciones.

Tampoco me convence pasarle los landmarks faciales crudos, ya que estaria exigiendole al modelo que tambien aprenda reconocimiento de emociones
Mi nueva propuesta:
- Un modelo **ya entrenado** o una RNN/transformer dedicado SOLO a interpretar la cara y pasar la **representación embebida** (no raw) al modelo principal.
## Obtencion de datos
- [CN Sordos](https://www.youtube.com/@CNSORDOSARGENTINA/videos)
- [Canal de asociacion civil](https://www.youtube.com/c/CanalesAsociaci%C3%B3nCivil/videos)
- [Videolibros en señas](https://www.videolibros.org/video/106)

## Resultados
