# Dataset augmentation usando PCA(incompleto)
La idea de este experimento es
1. Poner todos los samples de un mismo glifo(los landmarks, no el video en si) dentro de una matriz X
2. Realizar PCA sobre esa matriz y ver las principales direcciones en las que la seña varia entre señantes, quedarse con la de mayor variabilidad, la de mayor autovalor
3. Normalizar esos vectores para tener una representacion de la direccion en la que las señas varian y no en la magnitud
4. Realizar eso para cada uno de los glifos(no tienen que ser todos, solo los suficientes para ver)
5. Graficar un heatmap de la norma de las diferencias entre esos vectores de "principal variabilidad" entre señas, o agruparlos usando un algoritmo de clustering

Mi idea detras de esto es ver si hay direcciones de variabilidad comun no importa la seña(siempre se mueve el brazo un poco mas para arriba, un poco mas para abajo, etc) para luego poder aplicar esas direcciones de variabilidad "universales" a los datos y aumentar el tamaño del dataset. La idea del heatmap es que si 2 señas sufren de una variabilidad en la misma direccion(ignoro la escala al normalizarlo para simplificar el proceso), entonces la norma del vector resultante de la diferencia entre ellos deberia aproximarse a 0