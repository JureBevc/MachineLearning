import java.util.*;
import java.io.*;
import javax.imageio.*;
import java.awt.*;
import java.awt.image.*;
//
// The node
//

class Node{

  public double output = 0;
  public double weights[];
  public double deltaWeights[];
  public double grad = 0;

  int id;
  public Node(int numberOfInputs, int id){
    weights = new double[numberOfInputs];
    deltaWeights = new double[numberOfInputs];
    for(int i = 0; i < weights.length; i++)
      weights[i] = Math.random();
    this.id = id;
  }

  public void feed(Node[] lastLayer){
    // output = f(sum(input * weight))
    double sum = 0;
    for(int i = 0; i < lastLayer.length; i++){
      sum+=lastLayer[i].output * weights[i];
    }
    output = act(sum);
  }

  public void calcOutputGradient(double targetValue){
    double delta = targetValue - output;
    grad = delta * actDerivative(output);
  }

  public void calcHiddenGradient(Node[] nextLayer){
    double dow = 0;
    // Sum of the derivatives of weights of next layer
    for(int i = 0; i < nextLayer.length - 1; i++){
      dow += nextLayer[i].weights[id] * nextLayer[i].grad;
    }
    //
    grad = dow * actDerivative(output);
  }

  public void updateWeights(Node[] lastLayer){
    // eta -> learn reate
    // alpha -> momentum
    double eta = 0.01;
    double alpha = 0.5;
    //System.out.println("Grad: " + grad);
    for(int i = 0; i < lastLayer.length - 1; i++){
      double newDelta =
      eta
      * lastLayer[i].output
      * grad
      + alpha
      * deltaWeights[i];
      deltaWeights[i] = newDelta;
      weights[i] += deltaWeights[i];
    }
  }

  private static double act(double x){
    return Math.tanh(x);
  }

  private static double actDerivative(double x){
    double r = Math.tanh(x);
    return 1 - r * r;
  }

}

//
// The net
//

class Net{

  public Node nodes[][];

  public void createNet(int nodeNumbers[]){
    System.out.println("Creating " + nodeNumbers.length + " layers. (" + Arrays.toString(nodeNumbers) + ")");
    nodes = new Node[nodeNumbers.length][];
    for(int i = 0; i < nodeNumbers.length; i++){
      nodes[i] = new Node[nodeNumbers[i] + 1]; // +1 for biased nodes
      for(int j = 0; j < nodeNumbers[i] + 1; j++)
        nodes[i][j] = new Node((i == 0 ? 0 : nodeNumbers[i - 1] + 1), j);
      nodes[i][nodeNumbers[i]].output = 1; // biased node output
    }
  }

  public void feedInput(double inputValues[]){

    // Input layer
    if(inputValues.length != nodes[0].length - 1)
      System.out.println("Input values don't match the number of neurons!");

    for(int i = 0; i < inputValues.length; i++)
      nodes[0][i].output = inputValues[i];

    // Forward propagation
    for(int i = 1; i < nodes.length; i++){
      for(int j = 0; j < nodes[i].length - 1; j++){
        nodes[i][j].feed(nodes[i - 1]);
      }
    }
  }

  public void backProp(double targetValue[]){

    // Calculate error
    // RMS? -> sqrt(sum(delta^2) / n)
    double err = 0;
    for(int i = 0; i < targetValue.length; i++){
      double delta = targetValue[i] - nodes[nodes.length - 1][i].output;
      err += delta * delta;
    }
    err = Math.sqrt(err / targetValue.length); // RMS

    // Output layer gradient
    for(int i = 0; i < targetValue.length; i++){
      nodes[nodes.length - 1][i].calcOutputGradient(targetValue[i]);
    }

    // Hidden layer gradients
    for(int i = nodes.length - 2; i > 0; i--){
      for(int j = 0; j < nodes[i].length - 1; j++){
        nodes[i][j].calcHiddenGradient(nodes[i + 1]);
      }
    }

    // Update weights
    for(int i = nodes.length - 1; i > 0; i--){
      for(int j = 0; j < nodes[i].length - 1; j++){
        nodes[i][j].updateWeights(nodes[i - 1]);
      }
    }
  }

  public String getResult(){
    String r = "";
    for(int i = 0; i < nodes[nodes.length - 1].length - 1; i++)
      r += nodes[nodes.length - 1][i].output + " ";
    return r;
  }

  public double[] getAllResults(){
    double r[] = new double[nodes[nodes.length - 1].length - 1];
    for(int i = 0; i < nodes[nodes.length - 1].length - 1; i++)
      r[i] = nodes[nodes.length - 1][i].output;
    return r;
  }

}

//
// Main
//

public class Main{

  Net net;
  int passes = 0;
  File imgFolder;
  public void start(){
    // Create net
    net  = new Net();
    net.createNet(new int[]{100,50, 10});
    try{

      imgFolder = new File("tests/");
      System.out.println("Number of tests: " + imgFolder.listFiles().length);

      int initialLearningPasses = 500000;
      System.out.println("Initial learning passes = " + initialLearningPasses + "...");

    for(int i = 0; i < initialLearningPasses; i++){
      learn();
      if(i % 500 == 0)
        System.out.println(i + "...");
    }
    System.out.println("\nFinished.");
    System.out.flush();
    // User input
    Scanner sc = new Scanner(System.in);
    String line = sc.nextLine();
    while(!line.equals("q")){
      BufferedImage img = ImageIO.read(Main.class.getResourceAsStream("/usrtest/test.png"));
      guess(img);
      line = sc.nextLine();
    }
    sc.close();
    }catch(Exception e){
      e.printStackTrace();
    }
  }

public void learn(){
    try{
      double input[] = new double[100];
      double output[] = new double[10];

      // Pick image

      int fileNumber = (int)(Math.random() * imgFolder.listFiles().length);
      String fileName = imgFolder.listFiles()[fileNumber].getName();
      int number = Integer.parseInt("" + fileName.charAt(fileName.length() - 5));
      output[number] = 1;
      BufferedImage img = ImageIO.read(Main.class.getResourceAsStream("/tests/" + fileName));

      // Set inputs
      for(int i = 0; i < img.getWidth(); i++){
        for(int j = 0; j < img.getHeight(); j++){
          int b = img.getRGB(i, j) & 0XFF;
          input[i * img.getHeight() + j] = b / (255.0);
        }
      }
      net.feedInput(input);
      net.backProp(output);
      /*double[] r = net.getAllResults();
      for(int i = 0; i < r.length; i++)
        System.out.print((int)(net.getAllResults()[i] * 100)/100f + " ");
      System.out.println();*/
    }catch(Exception e){
      e.printStackTrace();
    }
  }

  public void guess(BufferedImage img){
    try{
      double input[] = new double[100];
      for(int i = 0; i < img.getWidth(); i++){
        for(int j = 0; j < img.getHeight(); j++){
          int b = img.getRGB(i, j) & 0XFF;
          input[i * img.getHeight() + j] = b / (255.0 * 2);
        }
      }
      net.feedInput(input);
      double[] r = net.getAllResults();
      int mi = 0; double m = r[0];
      for(int i = 0; i < r.length; i++)
        if(r[i] > m){
          m = r[i];
          mi = i;
        }
      System.out.println("My guess:");
      for(int i = 0; i < r.length; i++){
        if(i == mi)System.out.print("\u001B[32m");
        System.out.println(i + ": " + (int)(net.getAllResults()[i] * 100) + "%");
        System.out.print("\u001B[0m");
      }
      System.out.println();
    }catch(Exception e){
      e.printStackTrace();
    }
  }

  public static void main(String[] args){
   Main main = new Main();
   main.start();
  }

}
