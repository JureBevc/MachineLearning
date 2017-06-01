import java.util.*;
import java.io.*;

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
    return 1 - x * x;
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

}

//
// Main
//

public class Main{

  Net net;
  int passes = 0;
  public void start(){
    // Create net
    net  = new Net();
    net.createNet(new int[]{1,1});
    try{
      int initialLearningPasses = 1000;
      System.out.println("Initial learning passes = " + initialLearningPasses + "...");
    for(int i = 0; i < initialLearningPasses; i++)
      learn();
    System.out.println("\nFinished learning.");
    System.out.flush();
    System.out.println("...");
    System.out.flush();
    // User input
    Scanner sc = new Scanner(System.in);
    String line = "";
    do{
      line = sc.nextLine();
      if(line.equals("q"))
        continue;

      double input[] = new double[1];
      input[0] = Double.parseDouble(line.split(" ")[0].trim());
      net.feedInput(input);
      //net.backProp(output);
      System.out.println(net.getResult());
    }while(!line.equals("q"));
    sc.close();
    }catch(Exception e){
      e.printStackTrace();
    }
  }

  public void learn(){
    try{
    BufferedReader br = new BufferedReader(new FileReader("Inputs.txt"));
    double input[] = new double[1];
    double output[] = new double[1];
    while(br.ready()){
      String line = br.readLine();
      input[0] = Double.parseDouble(line.split(" ")[0].trim());
      output[0] = Double.parseDouble(line.split("=")[1].trim());
      net.feedInput(input);
      net.backProp(output);
    }
    passes++;
    br.close();
    }catch(Exception e){
      e.printStackTrace();
    }
  }

  public static void main(String[] args){
   Main main = new Main();
   main.start();
  }

}
