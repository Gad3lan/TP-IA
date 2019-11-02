package perceptron;

import java.util.Arrays;
import java.util.Random;

public class OnlinePerceptron {
	public static final int DIM = 3; // dimension de l'espace de representation
    public static float[] w = new float[DIM]; // parametres du modèle
    public static float[][] data = { // les observations
      {1,0,0}, {1,0,1} , {1,1,0},
      {1,1,1}
    };
    public static int[] refs = {-1, -1, -1, 1}; // les references

    public static void update(double[] w, float[] x, int y) {
    	for (int i = 0; i < w.length; i++) {
    		w[i] += y*x[i];
    	}
    }
    
    public static boolean wellClassed(double[] w, float[] x, int y) {
    	int tmp = 0;
    	for (int i = 0; i < x.length; i++) {
    		tmp += w[i] * x[i];
    	}
    	tmp /= y;
    	return (tmp <= 0);
    }
    
    public static boolean checkAll(double[] w) {
    	for (int i = 0; i < data.length; i++) {
    		if (wellClassed(w, data[i], refs[i])) {
    			return false;
    		}
    	}
    	return true;
    }
    
    public static int learn(double[] w) {
    	int epochNb = 0;
    	while(!checkAll(w)) {
    		for (int i = 0; i < data.length; i++) {
    			if (wellClassed(w, data[i], refs[i])) {
    				update(w, data[i], refs[i]);
    			}
    		}
    		epochNb++;
    	}
    	return epochNb;
    }
    
public static void main(String[] args) {	
	    // Exemple de boucle qui parcourt tous les exemples d'apprentissage
	    // pour en afficher a chaque fois l'observation et la reference.
    	for (int l = 0; l < 1; l++) {
	    	double[] test = {Math.random(), Math.random(), Math.random()};
	    	//System.out.println(Arrays.toString(test));
		    for (int i = 0; i < data.length; i++) {
		        float[] x = data[i];
		        //System.out.println("x= "+Arrays.toString(x)+ " / y = "+refs[i]);
		    }
		    int n = learn(test);
		   if (n != 11) {
		    	System.out.println(l + ": " + n);
	    	}
    	}
    	System.out.println("Finished");
    }
}
