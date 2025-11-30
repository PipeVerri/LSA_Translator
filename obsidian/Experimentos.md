Que tan posible es hacer traduccion en tiempo real de LSA a español?
# Investigacion previa y estado del arte
La primera corriente fue la que intente replicar yo. Usar una RNN entrenada para identificar glosas individuales, luego usar heurisiticos como aceleracion, ventanas de tamaño determinado, etc. para detectar cuando se hacia una glosa, y pasarsela a la RNN. El problema con esto es que es muy dificil recortar las señas asi por cosas como coarticulacion, variaciones de velocidad de una misma seña en una frase vs aislada. El principal problema era la segmentacion heuristica que andaba como el culo

Despues vinieron modelos gloss-free que intentaban hacer la segmentacion de glosas, la interpretacion textual de glosas, y la traduccion de glosas -> español en un unico transformer. Andaban bien, pero con pocos datos(como paso en LSA-T) andan mal porque tienen que aprender 3 tareas diferentes en una sola pasada.

Ahora lo nuevo que esta saliendo son sistemas semi-gloss free donde es todo esto que voy a hacer de la LLM. Es algo muy nuevo que salio este año pero parece ser lo que mejor anda
# Experimento 1
## Descripcion
- Una RNN simple de 5 capas ocultas, los videos los elegi procesar a 24fps. 
- Para el processado en tiempo real, lo procesaba en ventanas de 2 segundos.
- Usaba un umbral para el softmax para distinguir si hacia la seña o no estaba haciendo nada
### Resultados
Aunque distinguia las señas correctamente cuando se las hacia, y lograba tener muy buen validation accuracy al entrenarla(~97%), el problema era que el  metodo del umbral no servia para distinguir cuando no se estaban haciendo las señas. Eso es porque para 2 señas muy parecidas(rojo y amarillo por ejemplo), aunque el resultado fuera rojo, tenia accuracy bajo porque el amarillo tambien tenia mucho peso.

Lo que voy a hacer para resolver esto es:
- Bajar los FPS a ~6. Quiero que el procesamiento sea lo mas a corto plazo posible
- Cada seña dura ~2s, asi que si mis señas duran mas de 10 frames, voy a usar una LSTM o una GRU
- Voy a no solo entrenar el modelo con las señas de LSA64, tambien lo voy a entrenar con videos de oradores de charlas TED haciendo cosas con las manos pero no haciendo señas asi distingue entre sus opciones la opcion de no-seña

Otro problema que creo que pueda surgir es que como los de LSA64 estan todos sentados de la misma manera, siento que puede generalizar mal, para eso mi idea(futura) es:
- Tener una red dedicada a predecir si la seña es con una mano, o con las 2. Quizas probar la imagen con y sin espejar y comparar probabilidades para ver si la mano dominante es la zurda, o quizas es algo configurable
- Para la clasificacion de señas:
  - Una red dedicada a clasificar señas de 1 o 2 manos pero al pasarle una seña de una mano, poner en 0 la segunda mano y pasarle un flag de missing
  - 2 redes. Una dedicada a señas de una mano, la otra dedicada a señas de 2 manos

No se donde meteria el logit de "no seña", si en la primera o la segunda etapa.
# Experimento 2
## Descripcion
### Data processing
- 6fps
- Clips de charlas ted, 2s por clip.
- Para la escala de las manos, uso los puntos mano del pose
### Modelo
- Le agrege mas anchura, 300 en las capas ocultas
### Resultados
- Solo cuando pones las manos como orador toma como "no sign", esta cesgado porque son las charlas TED
- Aunque cuando repetis un monton la seña la toma bien, aun asi el problema son las ventanas intermedias entre no-seña y seña que no se ven como alguien con las manos abajo, y las toma como señas

Para trabajar con señas continuas, voy a tener que hacer que el modelo tambien aprenda segmentacion temporal ya que no hay forma de hacerla basandome en ventanas(la duracion de las señas varia demasiado).

**Puedo convertir los textos continuos de datasets como LSA-T en glosas usando un modelo de NLP, hacer segmentacion temporal con un modelo debilmente supervisado, y luego entrenar algo como mi idea original de la RNN con ese esquema**
https://arxiv.org/abs/2505.15438?utm_source=chatgpt.com

- Tambien quiero probar la tecnica basada en PCA para el aumento de datos, pero me pareceria innecesaria ya que el problema principal es la segmentacion, no el reconocimiento de glosas. Aun asi podria probarlo en una version chica y overfiteable de LSA-64, lo mismo con la idea de la RNN de 2 etapas.
- Tambien quiero probar con semi-supervised learning etiquetando a mano el LSA-T
# Experimento 3
Quiero probar usando LSA-T pero con 50-100 convoluciuones en vez de solo 1, usar pooling a lo largo del eje del tiempo, o ni siquiera usar convoluciones y pasarle los landmarks asi nomas. Me parece ridiculamente chico usar solo 3 numeros para representar la pose entera en un solo momento del tiempo. Tambien me parece que meterle informacion de la cara de forma raw no es buena idea, me parece que seria mejor usar un transformer o RNN que interprete la expresion de los landmarks y pasarle esa interpretacion al transformer dedicado a la traduccion.
