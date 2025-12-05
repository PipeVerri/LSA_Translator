# ¿Qué tan posible es hacer traducción en tiempo real de LSA a español?

En este documento voy a documentar mi proceso de pensamiento, qué probé, qué salió mal, y como tengo planeado seguir

---

# Investigación previa y estado del arte

La primera corriente (la que intenté replicar yo al principio) fue:

1. **Entrenar una RNN para identificar glosas individuales** a partir de un video ya segmentado.
2. Hacer **segmentación heurística** en tiempo real: mirar aceleración, ventanas de tamaño fijo, etc. para decidir cuándo alguien “hizo una seña”.
3. Cada vez que las heurísticas dicen “acá hubo una seña”, recorto el pedazo y se lo paso a la RNN.

El problema de este enfoque es que la segmentación heurística **se dejo de lado porque anda muy mal**. Algunos problemas son:
1. Hay coarticulacion: Si hago la seña de yo("me apunto con el dedo indice") y luego la de voy(apunto hacia adelante con el indice):
   - Si las grabara individualmente, el brazo iria desde abajo hacia apuntarme para "yo", o iria desde abajo hacia adelante para "voy"
   - Al hacerlas juntas, primero me apunto, y luego apunto para adelante **sin llegar a bajar el brazo** o a pausar.
2. La velocidad de una misma seña cambia dentro de una frase vs aislada, o hasta entre frases puede haber una gran diferencia(dependiendo de la velocidad del señante)

Después vinieron los **modelos gloss-free**:

- Hacen *todo* en un solo transformer:
  - segmentación de glosas,
  - interpretación textual de las glosas,
  - traducción glosas → español.
- Andan bien cuando tenés muchos datos, pero con datasets chicos (como LSA-T) funcionan peor porque les estás pidiendo aprender **tres tareas distintas a la vez** con poca señal.

Lo más nuevo que está saliendo son cosas **semi-gloss-free**, que se parecen más a lo que quiero hacer con una LLM(explicados posteriormente):
S
- No dependen tanto de anotaciones perfectas de glosas.
- Se apoyan en modelos de lenguaje grandes para la parte textual.
- Dejan a los modelos de visión/secuencia concentrarse más en la parte de “qué está pasando en el video”.

---

# Resumen de experimentos

- **Experimento 1:** RNN simple sobre LSA64 → buena clasificación aislada, mala en continuo por falta de segmentación.
- **Experimento 2:** RNN más grande + datos de “no-seña” → mejora, pero sigue fallando la segmentación temporal real.
- **Experimento 3:** Plan para pasar a LSA-T y modelos semi-gloss-free o gloss-free (modificando la arquitectura original).
---
# Cosas para probar

- Aumento de datos por **PCA** y **movimiento kinematico**
- Modelo en dos etapas:
  - detectar tipo de seña / no-seña,
  - luego clasificar seña específica.
- Semi-supervised:
  - etiquetar fragmentos de LSA-T,
  - mejorar alineaciones.
# Atribuciones
Este repositorio usa codigo de [LSA-T](https://github.com/midusi/LSA-T)