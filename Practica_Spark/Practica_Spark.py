#!/usr/bin/env python
# coding: utf-8

# # PRACTICA SPARK - Julia Hernández Elena

# In[1]:


import os
import pandas as pd

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import types

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window


from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder


# In[2]:


conf = (

    SparkConf()
    .setAppName(u"[ICAI] Ejercicios Spark")
    .set("spark.executor.memory", "7g")
    .set("spark.executor.cores", "5")
    .set("spark.default.parallelism", 600)
    .set("spark.sql.shuffle.partitions", 600) 
    .set("spark.dynamicAllocation.maxExecutors", 4) 
)


# In[3]:


spark = (

    SparkSession.builder
    .config(conf=conf)
    .enableHiveSupport()
    .getOrCreate()

)


# ### Leer ambos ficheros a Spark DataFrame y hacer las transformaciones necesarias para conseguir los esquemas establecidos:

# Creamos el esquema que queremos para leer el CSV de info_contenidos

# In[4]:


esquema_contenidos = T.StructType([

    T.StructField("id_contenido",T.IntegerType()),
    T.StructField("genero",T.StringType()),
    T.StructField("subgenero",T.StringType()),
    T.StructField("tipo_contenido",T.StringType())

])


# In[5]:


contenidos = (

    spark.read
    .options(header=False) #comprobamos que no tiene header
    .schema(esquema_contenidos)
    .csv('/datos/practica_spark/info_contenidos.csv') 

).cache()


# In[ ]:


print("#Esquema del dataframe de contenidos:")


# In[6]:


contenidos.printSchema()


# Los ficheros parquet tienen su propio esquema contenido en el fichero por lo que no lo podemos alterar hasta que lo cargemos: 

# In[6]:


audiencias = spark.read.load('/datos/practica_spark/muestra_audiencias.parquet').cache()


# In[ ]:


#audiencias.printSchema() #no imprimimos este por pantalla


# Cambiamos los tipos de las columnas y con un select cambiamos el orden de las columnas en el dataframe:

# In[7]:


audiencias = (
    audiencias
    .withColumn("id_contenido", audiencias["id_contenido"].cast("integer"))
    .withColumn("co_cadena", audiencias["co_cadena"].cast("integer"))
    .withColumn("it_inicio_titulo", audiencias["it_inicio_titulo"].cast("string"))
    .withColumn("it_fin_titulo", audiencias["it_fin_titulo"].cast("string"))
    .withColumn("it_inicio_visionado", audiencias["it_inicio_visionado"].cast("string"))
    .withColumn("it_fin_visionado", audiencias["it_fin_visionado"].cast("string"))
    
    .select('id_contenido', 'co_cadena', 'it_inicio_titulo', 'it_fin_titulo', 'it_inicio_visionado','it_fin_visionado')
)


# In[ ]:


print("#Esquema del dataframe de audiencias tras manipularlo para conseguir el deseado:")


# In[10]:


audiencias.printSchema()


# ### ¿Cuántos registros tiene cada DF? 

# In[11]:


count_aud = audiencias.count()


# In[12]:


count_cont = contenidos.count()


# In[13]:


print("#Registros de muestras_audiencias: {}" .format(count_aud))
print("#Registros de info_contenidos: {}" .format(count_cont))
print("\n")


# ### ¿Cuántos contenidos (id_contenido) únicos tiene cada uno? ¿Cuántas cadenas (co_cadena) hay en el primer DF?

# In[21]:


contenido_unico_aud = (
    audiencias
    .select(F.countDistinct('id_contenido').alias('Contenidos unicos en muestras_audiencias'))
    .show()
)


# In[19]:


contenido_unico_cont = (
    contenidos
    .select(F.countDistinct('id_contenido').alias('Contenidos unicos en info_contenidos'))
    .show()
)


# In[20]:


cadenas_aud = (
    audiencias
    .select(F.countDistinct('co_cadena').alias('Cadenas en muestras_audiencia'))
    .show()
)


# ### En el primer dataset las columnas que marcan momentos (las variables que empiezan por it) están definidas como string, convertirlas a formato timestamp:

# In[8]:


audiencias = (
    audiencias
    .withColumn("it_inicio_titulo", audiencias["it_inicio_titulo"].cast("timestamp"))
    .withColumn("it_fin_titulo", audiencias["it_fin_titulo"].cast("timestamp"))
    .withColumn("it_inicio_visionado", audiencias["it_inicio_visionado"].cast("timestamp"))
    .withColumn("it_fin_visionado", audiencias["it_fin_visionado"].cast("timestamp"))
    
)


# In[ ]:


print("#Esquema del dataframe de audiencias con variables timestamp:")


# In[23]:


audiencias.printSchema()


# ### Calcular ahora la duración en minutos de cada programa usando it_inicio_titulo y it_fin_titulo, una vez calculado descartar del dataset todos los registros donde la duración sea menor que un minuto.

# Convertimos el formato timestamp a unix timestamp (en segundos), calculamos la diferencia y lo dividimos entre 60. Ademas redondeamos hasta 4 decimales porque no necesitamos tanta informacion.

# In[9]:


audiencias = (
    
    audiencias
    
    .withColumn(
    "duracion_contenido", 
    F.round((F.col("it_fin_titulo").cast("long") - F.col("it_inicio_titulo").cast("long"))/60, 4))

    .filter('duracion_contenido>=1')

)


# ### ¿Cuál es la duración media de los contenidos?

# In[25]:


Duracion_media = (
    audiencias
    .select(F.mean('duracion_contenido').alias("Duracion media de los contenidos en min"))
    .show()
)
    


# ### Filtrar ahora todos los registros donde el inicio de visionado sea después del fin de visionado ya que se entienden que estos registros son erróneos.

# In[10]:


audiencias = (
    audiencias
    .filter('it_fin_visionado>=it_inicio_visionado')
)


# ### Calcular la duración de cada visualización en minutos

# In[11]:


audiencias = (
    audiencias
    .withColumn(
    "duracion_visualizacion", 
    F.round((F.col("it_fin_visionado").cast("long") - F.col("it_inicio_visionado").cast("long"))/60, 4))
)


# ### Para cada contenido, cadena e inicio del título agregar (sumar) todos los minutos vistos

# In[12]:


agregado = (
    audiencias
    .groupBy('id_contenido', 'co_cadena', 'it_inicio_titulo')
    .agg(F.sum('duracion_visualizacion').alias('minutos_visualizados'))
)


# ### Guardar el nuevo DF con la información agregada en el esquema personal de hive con el nombre practica_audiencias_agregado (activar el modo overwrite)

# In[31]:


mi_user = os.environ.get('USER') #cogemos nuetro nombre de usuario
table_name = mi_user + ".practica_audiencias_agregado" #nombramos la tabla


# In[32]:


agregado.write.mode('overwrite').saveAsTable(table_name)


# ### Usando el DF con información extra de los contenidos, conseguir/pegar en cada registro del DF agregado los campos: genero, subgenero y tipo_contenido usando para ello el campo en común id_contenido

# In[13]:


df = (
    agregado
    #Hago un Join con la duracion del contenido
    .join(contenidos, 'id_contenido')
).cache()


# ### Construir ahora las siguientes columnas sobre el nuevo DF:
# 
# - ### id_weekday: Día de la semana (en literal) del inicio del programa.
# - ### id_month: Mes (en literal) del inicio del programa.
# - ### id_inicio: Hora del inicio del programa.

# In[14]:


df = (
    df
    
    #Creamos las nuevas columnas
    .withColumn("id_weekday", F.date_format(F.col("it_inicio_titulo"), "EEEE"))
    .withColumn("id_month", F.date_format(F.col("it_inicio_titulo"), "MMMM"))
    .withColumn("id_inicio", F.hour(F.col("it_inicio_titulo")))
    .drop(F.col("it_inicio_titulo"))
    
    
    #Cambiamos el nombre de la columna minutos_visualizados para conseguir el esquema dado
    .withColumn("minutos", df["minutos_visualizados"])
    
    #Seleccionamos el orden de las columnas en el orden dado en el esquema
    .select('id_contenido', 'minutos', 'co_cadena', 'genero', 'subgenero', 'tipo_contenido','id_weekday', 'id_month', 'id_inicio')

)


# In[ ]:


print("#Esquema del dataframe de contenidos y audiencias agragegado con las nuevas columnas id_weekday, id_month e id_inicio:")


# In[35]:


df.printSchema()


# In[36]:


df.limit(10).toPandas()


# ### Con el DF generado hasta ahora, vamos a entrenar un modelo de Machine Learning, queremos explicar los minutos visualizados en función del resto de variables:

# ##### (NOTA: No se usará la variable id_contenido ya que es un identificador)
# 

# In[15]:


df = df.drop('id_contenido')


# - ##### Usando StringIndexer codificar cada una de las columnas en numérica.

# In[16]:


columnas_quiero = ['co_cadena','genero', 'subgenero', 'tipo_contenido', 'id_weekday', 'id_month','id_inicio']

indexers = [ 

    StringIndexer(inputCol=i, outputCol=i + "_Index").setHandleInvalid("skip")  
    for i in columnas_quiero

]


# - ##### Usar OneHotEncoder en cada columna para generar las variables binarias.

# In[17]:


encoders = [ 

    OneHotEncoder(dropLast=False, inputCol=indexer.getOutputCol(), outputCol=indexer.getOutputCol() + "_encoded") 
    for indexer in indexers

]


# - ##### Usar VectorAssembler para conseguir un vector de features con todas las variables binarias.

# In[18]:


vectorAssembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol="features") 


# - ##### Encapsular todas las transformaciones de ML anteriores en una pipeline de Spark ML.

# In[19]:


pipeline = Pipeline(stages=indexers+encoders+[vectorAssembler]) 


# In[20]:


model = pipeline.fit(df)


# In[21]:


transformed = model.transform(df)


# - ##### ¿Qué tamaño tiene este vector de features?

# In[54]:


featues_length = len(transformed.first()["features"])


# In[55]:


print("#Tamanio del vector features: {}" .format(featues_length))
print("\n")


# - ##### Dividir xtrain en 80% Train y 20% Test.

# In[22]:


seed = 1234
trainDF, testDF = transformed.randomSplit([0.8, 0.2], seed=seed)
trainDF.cache()
testDF.cache()


# - ##### Entrenar un árbol de regresión para predecir la variable minutos.

# In[57]:


dt = DecisionTreeRegressor(labelCol='minutos') #toma como inputCo "features" de manera predeterminada


# In[58]:


model=dt.fit(trainDF)


# - ##### Evaluar el modelo resultante usando RMSE tanto en la muestra de entrenamiento como en la muestra de test: Comentar el resultado.

# In[59]:


predictionDF = model.transform(testDF)


# In[23]:


evaluator = RegressionEvaluator(labelCol="minutos")


# In[61]:


rmse_train = evaluator.evaluate(model.transform(trainDF))
rmse_valid = evaluator.evaluate(predictionDF)


# In[62]:


print("#RMSE de trainDF: {:3f}".format(rmse_train))
print("#RMSE de testDF: {:3f}".format(rmse_valid))


# In[64]:


print("En primer lugar podemos comentar que el error de test no es mucho mayor que el error de entrenamiento por lo que podriamos decir" 
print("que nuestro modelo no sobre-entrena lo que es positivo.") 
print("Sin embargo, tambien podemos comentar que un error de 718 minutos que es un errror bastante grande. Nuestro modelo no aproximamuy bien los") 
print("minutos de visualizacion. Quiza necestirariamos un modelo mas complejo como un Random Forest que consiguiera predecir mejor esta variable")


# #### *OPCIONAL:*
# - ##### Entrenar también un modelo de Gradient_boosting (implementado en Spark en el objeto GBTRegressor).
# - ##### Usar validación cruzada sobre la muestra de entrenamiento para tunear los hiperparámetros convenientes hasta conseguir el mejor resultado que seamos capaces.

# In[24]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# Creamos el Gradient Boosting que aplicaremos a la columna de features calculada anteriormente mendiante StringIndexer, OneHotEnconder y VectorAssembler:

# In[ ]:


gbt = GBTRegressor(labelCol="minutos", seed=2019)


# Probamos a tunear varios parametros:

# In[27]:


paramGrid = (

    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [2, 4, 6, 8])
    .addGrid(gbt.maxIter, [10, 50, 100, 150])
    .build()

)


# Usando Cross Validation ajustamos el modelo a la columna de features y los hiper-parametros del mismo:

# In[28]:


crossval = CrossValidator(

    estimator = gbt, #aqui ya emos metido que queremos el gbt sin parametros
    estimatorParamMaps = paramGrid, #aqui metemos los parametros
    evaluator = evaluator,
    numFolds = 2

)


# In[ ]:


cvGTBModel = crossval.fit(trainDF) #coge el data-set de entrenamiento


# In[ ]:


cvModel.avgMetrics #media en cada uno de esos puntos


# In[ ]:


mejor = np.argsort(cvModel.avgMetrics)[0]


# Comprobamos cuales son los parametros para el mejor resultado de RMSE. Si fueran extremos del ParamGrid ajustariamos la maya:

# In[ ]:


cvModel.avgMetrics[mejor]


# In[ ]:


cvModel.getEstimatorParamMaps()[mejor] #nos quedamos con la configuracion de la maya para la mejor configuracion


# In[ ]:


rmse_train = evaluator.evaluate(cvModel.transform(trainDF))
rmse_valid = evaluator.evaluate(cvModel.transform(testDF))


# In[ ]:


print("## RMSE (Train) de Cross Validation del modelo Gradient Boosting: {:.3f}".format(rmse_train))
print("## RMSE (Valid) de Cross Validation del modelo Gradient Boosting: {:.3f}".format(rmse_valid))


# In[ ]:


print("Como podemos comprobar obtenemos bastante mejor resultado que en el arbol de decision.") 
print("La tecnica de validacion cruzada y este modelo mas complejo han dado resultado")


# In[315]:


spark.stop()


# 
# **NOTA:**
# Para guardar el .ipyng a un.py usamos:
# >export PATH=/opt/share/anaconda3/bin:$PATH
# 
# >jupyter nbconvert --to python jupyter_code/Practica_Spark.ipynb
# 
# Para lanzarlo desde consola y guardar el output en un .txt:
# >spark2-submit jupyter_code/Practica_Spark.py 2 > jupyter_code/Prints_PracticaSpark.txt
# 
# 

# In[ ]:




