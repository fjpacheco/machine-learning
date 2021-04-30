## **Objetivos e Introduccion**

La agencia quiere utilizar la información recolectada para dirigir campañas de recaudación de impuestos y poder dirigir a los fiuagentes recaudadores a inspeccionar.
* ¿O sea, encontrar algun patron en el cual habria que decirles a los agentes para que vayan a inspeccionar segun ese patron? 
    * Por ejemplo, gente con pocas horas trabajadas pero declararon alto ingreso.. ahi les decimos que ese es factor importante para mandar un fiuagente (?

## **Ordenamiento del Notebook**

* el notebook esté ordenado de forma que cuente una historia

* que contenga texto e imágenes que expliquen cuál fue el proceso de análisis que se fue haciendo.

## **Orden de las preguntas a la hora de trabajar**

1. ¿Cuáles fueron las preguntas que se plantearon inicialmente?
2. ¿Qué se hizo para responder a esas preguntas?
3. De los gráficos y análisis hechos, ¿qué conclusiones se pueden sacar?
4. **A partir del trabajo en los anteriores puntos, ¿surgieron nuevas dudas? -> Volver al paso 2´**
5. A partir de todo el análisis anterior, construir el código baseline que se va a usar para la primera ronda de campaña digital. Fundamentar el código basándose en las conclusiones de los anteriores puntos.

Se espera que presenten el notebook en una charla de 20 minutos y dijeron que quieren explícitamente el formato:
- pregunta
- gráfico/s para responder esa pregunta
- por cada gráfico quieren un comentario escrito de que se interpreta en el mismo
- respuesta a la pregunta en base a todos los gráficos y análisis de los mismos

## **Datos recolectados**

**edad**: número, indica la edad   
**categoria_de_trabajo**: texto, cada valor indica el tipo de trabajo  
**educacion_alcanzada**: texto, cada valor indica la educación alcanzada  
**anios_estudiados**: número, indica la cantidad de años que estudió  
**estado_marital**: texto, indica el estado marital  
**trabajo**: texto, indica que tipo de trabajo realiza  
**rol_familiar_registrado**: texto, indica que rol tiene dentro del grupo familiar  
**religion**: texto, indica a que religión pertenece  
**genero**: texto, indica género  
**horas_trabajo_registradas**: número, indica la cantidad de horas que trabaja por semana  
**barrio**: texto, indica que barrio de capital reside  
**tiene_alto_valor_adquisitivo**: número (variable target) indica si tiene alto valor adquisitivo (valor 1) o bajo valor adquisitivo (valor 0)  
**ganancia_perdida_declarada_bolsa_argentina**: número, indica el resultado de las operaciones en bolsa que realizó durante el último año.  

---

## <span style="color:teal; font-weight: bold;">¿Preguntas que se plantearon inicialmente?</span> 

Aca debemos debatir cual hacernos!

Enlisto varias que se me ocurrieron, pero no se si van por ese lado...

## <span style="color:teal; font-weight: bold;">Muchas preguntas anotadas</span> 

Preguntas que pensé, teniendo en cuenta relación de entre los datos. Hay que debatirlas!

* **¿Presentar Barrio y Proporcion de personas que declararon un alto u bajo ingreso?**
    * ¿Estaria piola empezar poniendo un mapa de CABA? ¿Mostrar participación por comunas?
    * ¿Tiene alto u bajo ingreso segun el barrio en el que se encuentra?
    
* **¿Presentar la gente que haya declarado en la Bolsa Argentina?**
    * ¿Ver si hay una relacion con el barrio y la declaracion de la bolsa? 
    * ¿Ver si hay relacion con declaracion en la bolsa y con la decleración de si tiene alto u bajo ingreso?
    * ¿Y la relacion de bolsa declarada, barrio y sobre si tiene bajo u alto ingreso? ¿Heatmap o algun grafico para relacioanr esas 3 variables?

* **¿Presentar Edades y Generos encuestadas?**
    * ¿Relacion entre edad con la declaracion del genero? 
     * ¿Hay grupo de edad predominante que responde la encuesta y otros no se registró datos?
        * ¿El genero afecta en ese grupo predominante? ¿O es equitativa?
            * En otras palabras.. ¿Responden mas las mujeres de cierta edad, o los hombres de cierta edad?
    * ¿Influye la edad si se declara bajo u alto ingreso?
        * ¿Hay un rango de edad que muestre predominancia en mayor u bajo ingreso?
    * ¿Influye el genero si se declara bajo u alto ingreso?
        * Para cada genero por individual, ¿hay un rango de edad que muestre predominancia en mayor u bajo ingreso? Ejemplo, mujeres en tal edad hay mayor ingreso, idem hombres.
    * ¿Relacion entre las 3 simultaneamente: alto ingreso, genero y edad? 
    * ¿Y la bolsa declarada? ¿Comentamos algo al respecto o ya seria mucho?
    

* **¿Presentar Trabajo, Categoria de Trabajo y Horas trabajadas?** O sea, 3 variables recontra relacionadas.
    * ¿Influye el tipo de trabajo en la declaracion de alto u bajo ingreso?
        * ¿Y las horas trabajadas en ese trabajo?
            * ¿Cuanto mayor horas trabajads, mayor declaracion de alto ingreso?
        * ¿Y segun el tipo de categoria? 
            * ¿Hay una categoria predominante respecto al trabajo que declaró tener alto ingreso?
    * ¿Hay una relacion con respecto a la edad?
        * ¿En cierta edad, con cierto trabajo o cierta cantidad de horas o cierta categoria, afecta a la declaracion de tener algo ingreso u bajo?
        * Se me ocurre por ejemplo... ¿Cuanto mas joven sos, menor horas trabajas, se declaro alto u bajo ingreso? 
            * O sea.. hay una categoria de edad especial el cual se labura mucho o poco y se declara que tiene alto u bajo ingreso?
            

* **¿Educación alcanzada y años estudiados?** 
    * ¿Hay relacio entre educacion alcanzada y años estudiados?
    * ¿Influye tener mas nivel de ecucacion con la declaracion de alto u bajo ingreso?
    * ¿Influye la cantidad años estudiados con la declaracion de alto u bajo ingreso?
    * ¿Y la edad acá? ¿Mayor años estudiados, mayor educacion, mayor ingresos.. dependerá la edad? ¿Quizas para menor edad, o mayor edad?
 
* **¿Presentar Estado marital y rol familiar registrado?**
    * ¿Influye estado marital con la declaracion de alto u bajo ingreso?
        * ¿Y el rol familiar dentro de ese estado marital?

* **¿Religion?**
    * ¿Influye religion con la declaracion de alto u bajo ingreso?

