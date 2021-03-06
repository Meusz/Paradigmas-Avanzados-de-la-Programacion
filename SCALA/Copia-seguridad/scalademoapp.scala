import scala.io.StdIn.readLine

object Ejercicio extends App {
  
  val r = scala.util.Random
  
  val X = 7
  val Y = 9
  
  /*Crea un tablero totalmente aleatorio*/
  def Generar_Tablero(n:Int): List[Int]={
    if(n == 1)
      List(1+r.nextInt(6))
    else
      List(1+r.nextInt(6)) ::: Generar_Tablero(n-1)
  } 
  /*Permite obtener una matriz en la consola*/
    def Imprimir_matriz_XxY(x_aux:Int,y_aux:Int,l:List[Int]): Unit ={
      if ( y_aux != Y){
        print( l(y_aux*X+x_aux) +"\t")
        if(x_aux == X-1){
          println()
          Imprimir_matriz_XxY(0,y_aux+1,l)
        }
        else 
          Imprimir_matriz_XxY(x_aux+1,y_aux,l)
      } 
    }
    /*Comprueba que dos elementos son adyacentes*/
    def son_adyacentes(x_n1:Int,y_n1:Int,x_n2:Int,y_n2:Int):Boolean={
       if(x_n1 == x_n2 && (y_n1 == y_n2 +1 || y_n1 == y_n2 - 1))
         true
       else if(y_n1 == y_n2 && (x_n1 == x_n2 +1 || x_n1 == x_n2 - 1))
         true
       else 
         false
    }
    
    /* Cambia dos elementos de ubicacion*/
    def mover(n1:Int,n2:Int,n_aux:Int, lista:List[Int]):List[Int]= {
      if(n_aux == Y*X-1){
        if(n_aux == n1)
          List(lista(n2))
        else if(n_aux == n2)
          List(lista(n1))
        else
          List(lista(n_aux))
      }
      else{
        if(n_aux == n1)
          List(lista(n2)) ::: mover(n1,n2,n_aux+1,lista)
        else if(n_aux == n2)
          List(lista(n1)) ::: mover(n1,n2,n_aux+1,lista)
        else
          List(lista(n_aux)) ::: mover(n1,n2,n_aux+1,lista)
      }
    }
    /*Cuantos diamantes estan combinados de manera horizontal*/
    def combinacion_cuantas_fichas(n_aux:Int, lista:List[Int]):Int= {
      if(n_aux/X == (n_aux+1)/X && lista(n_aux) == lista(n_aux+1))
        1 + combinacion_cuantas_fichas(n_aux+1, lista)
      else
        1
    }
    
    /*Comprueba que hay diamantes combinados de manera horizontal*/
    def Hay_combinacion(n_aux:Int, lista:List[Int]):Boolean= {
      if(n_aux == Y*X-1)
        false
      else{
        if(combinacion_cuantas_fichas(n_aux,lista) > 2)
          true
        else
          Hay_combinacion(n_aux+1, lista)
      }
    }
    /*--------------------------Eliminar Gemas--------------------------*/
    def lista_vacia(n:Int):List[Int]={
      if(n==1)
        List(0)
      else
        List(0) ::: lista_vacia(n-1)
    }
    
    def combo_de_gemas(tablero:List[Int],n:Int):Int ={
      if( n == X*Y-1 )
        1
      else if( n/X != (n+1)/X && tablero(n).equals(tablero(n+1)))
        1
      else if( n/X == (n+1)/X && tablero(n).equals(tablero(n+1)) )
        1 + combo_de_gemas(tablero,n+1)
      else
        1
    }
    
    def eliminar_gemas(tablero:List[Int],n:Int):List[Int] ={
      if(n == X*Y-1)
        List(tablero(n))
      else{
        val combo = combo_de_gemas(tablero, n)
        if( combo >= 3 && n+combo >= X*Y-1 )
          lista_vacia(combo) 
        else if( combo >= 3 )
          lista_vacia(combo) ::: eliminar_gemas(tablero,n+combo)
        else 
          List(tablero(n)) ::: eliminar_gemas(tablero,n+1)
      }      
    }
    
    /*---------------------------------Caer Gemas---------------------------------*/
    def debe_caer(tablero:List[Int],n:Int):Boolean ={
      if(tablero(n) == 0)
        true
      else if( n == X*Y-1 )
        false
      else{
        debe_caer(tablero,n+1)
      }
    }
    
    def caer_columna(tablero:List[Int],col:Int,col_aux:Int,fil_aux:Int,caer:Boolean):List[Int]={
     
      if(col_aux == col && !caer && tablero(fil_aux * X +col_aux)== 0)
        caer_columna(tablero,col,col_aux,fil_aux,true)
      else{
         //println( fil_aux + " " +col_aux + "  " + caer)
        if(fil_aux == 0 ){
          if( col_aux == 0){
             if(caer && col_aux == col)
                List(1+r.nextInt(6))
              else{
                List(tablero(fil_aux*X+col_aux))
              }
          }
          else {
              if(caer && col_aux == col)
                caer_columna(tablero,col,col_aux-1,fil_aux,caer) ::: List(1+r.nextInt(6))
              else
                caer_columna(tablero,col,col_aux-1,fil_aux,caer) ::: List(tablero(fil_aux*X+col_aux))
          }
       }else{
         
              if(col_aux == 0){
                if( caer && col_aux == col)
                   caer_columna(tablero,col,X-1,fil_aux-1,caer) ::: List(tablero((fil_aux-1)*X+col_aux)) 
                else
                  caer_columna(tablero,col,X-1,fil_aux-1,caer) ::: List(tablero(fil_aux*X+col_aux))
              }
              else{
         
                if( caer && col_aux == col)
                   caer_columna(tablero,col,col_aux-1,fil_aux,caer) ::: List(tablero((fil_aux-1)*X+col_aux))
                else
                  caer_columna(tablero,col,col_aux-1,fil_aux,caer) ::: List(tablero(fil_aux*X+col_aux)) 
              }
          }
      }
    }
    
    
    
 
    
    def caer_gemas(tablero:List[Int],col:Int):List[Int]={
      if(col ==0){
         val fin = caer_columna(tablero, col, X-1, Y-1, false)
         if(debe_caer(fin,0))
           caer_gemas(fin,X-1)
         else
           fin
      }
      else{
        val modif = caer_columna(tablero, col, X-1, Y-1, false)
        caer_gemas(modif,col-1)
      }
    }
    
    
    /*--------------------------------Bucle de Juego -------------------------------*/
    
    def jugar(tablero:List[Int]): List[Int]={
        print("Posicion x del primer valor a cambiar: ")
        val x_n1 = readLine.toInt - 1 
        print("Posicion y del primer valor a cambiar: ")
        val y_n1 = readLine.toInt - 1
        print("Posicion x del segundo valor a cambiar: ")
        val x_n2 = readLine.toInt - 1
        print("Posicion y del segundo valor a cambiar: ")
        val y_n2 = readLine.toInt - 1
        
        if(son_adyacentes(x_n1,y_n1,x_n2,y_n2)){
          val movido = mover(y_n1*X+x_n1,y_n2*X+x_n2,0,tablero)
          //eliminar
          val eliminar = eliminar_gemas(movido,0)
          //crear_fichas y caer existentes
          caer_gemas(eliminar,X-1)
          
        }
            
            
        else{
          println("Los valores ingresados no son adyacentes. Debes repetir el proceso: ")
          jugar(tablero)
        }
        
    }
    
    
    def bucle_juego(tablero:List[Int],turno:Int):Unit={
      println("Estas jugando en el turno "+turno+"\nEste es el estado de tu juego: \n")
      Imprimir_matriz_XxY(0,0,tablero)
      print("\nEs hora de seleccionar que quieres hacer:\n?Seguir judando? y/n  ")
      val seguir = readLine().toString()
      if( seguir.equals("y")){
        println("\nSi desea continuar debe ingresar que dos posiciones desea intercambiar  ")
        val new_tablero = jugar(tablero)
        
        bucle_juego(new_tablero,turno+1)
      }
      else if(seguir.equals("n"))
        println("Finalizo el turno")
      else{
        println("No se reconocio su respuesta. Se procede a resetear el turno")
        bucle_juego(tablero,turno) 
      }               
    }
    
    
    
    def main(): Unit ={
      /*MAIN*/
      println(X*Y-1)
      println("---------------------BIENVENIDO A JEWELS HERO----------------------\n")
      println("Se va a establecer la dificultad 1 de juego.\n\nCreando tablero 7x9 con diamantes\n\n")
      val tablero = Generar_Tablero(X*Y) //Inicializamos la lista
      bucle_juego(tablero,0)
    }
    
    main()
  
}