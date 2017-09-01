import java.io.*;
public class Main{

 public static void main(String[] args){

  try{
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("besede.txt"), "UTF-8"));
    FileWriter fw = new FileWriter(new File("slovensko.txt"));
    while(br.ready()){
      String line = br.readLine();
      if(line.length() == 5 && !(line.contains("č") || line.contains("š") || line.contains("ž"))){
        line = line.replaceAll("á", "a").replaceAll("é", "e").replaceAll("í", "i").replaceAll("ó", "o").replaceAll("ú", "u");
        fw.write(line + "\n");
      }
    }
    fw.close();
    br.close();
  }catch(Exception e){
      e.printStackTrace();
  }


 }

}
