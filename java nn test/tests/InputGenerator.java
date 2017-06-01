import java.io.*;
public class InputGenerator{

  public static void main(String[] args){
    try{
    FileWriter fw = new FileWriter(new File("Inputs.txt"));
    for(int i = 0; i < 2000; i++){
      double n1 = (Math.random() * 2 * Math.PI);
      fw.write(n1 + " = " + Math.sin(n1) + "\n");
    }
    fw.close();
  }catch(Exception e){
    e.printStackTrace();
  }
  }

}
