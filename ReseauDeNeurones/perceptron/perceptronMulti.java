package perceptron;

import java.util.Arrays;
import java.util.Random;

import mnisttools.MnistReader;

public class perceptronMulti {
	
	public static int seed = 1234;
    public static Random GenRdm = new Random(seed);//Random(seed);

	public static int[] OneHot(int tag) {
		int[] v = new int[10];
		for (int i = 0; i < 10; i++) {
			v[i] = 0;
		}
		v[tag] = 1;
		return v;
	}
	
	public static double dot(float[][] w, float[] x, int l) {
		double tot = 0;
		for (int i = 1; i < x.length; i++) {
			tot += x[i]*w[i][l];
		}
		return tot + w[0][l];
	}
	
	public static float[] InfPerceptron(float[][] w, float[] x) {
		double[] dProbs = new double[w[0].length];
		float[] fProbs = new float[w[0].length];
		float tot = 0;
		for (int i = 0; i < w[0].length; i++) {
			double z = Math.exp(dot(w, x, i)); 
			dProbs[i] = z;
			tot += z;
		}
		
		for (int i = 0; i < w[0].length; i++) {
			fProbs[i] = (float) (dProbs[i]/tot);
		}
		//System.out.println(Arrays.toString(dProbs) + ", " + tot);
		//System.out.pr
		return fProbs;
	}
	
	public static float[][] InitialiseWeights(int DIM) {
    	float[][] w = new float[DIM][10];
    	float alpha = 1.0f/DIM;
    	for (int i = 0; i < w.length; i++) {
    		for (int l = 0; l < w[i].length; l++) { 
    			w[i][l] = alpha*(GenRdm.nextFloat()-0.5f);
    		}	
    	}
    	return w;
    }
	
	public static void updateWeights(float[] x, float[] y, int[] p, float eta) {
		for (int l = 0; l < ImageOnlinePerceptronMulti.w[0].length; l++) {
			ImageOnlinePerceptronMulti.w[0][l] -= eta*(y[l]-p[l]);
		}
		for (int i = 1; i < ImageOnlinePerceptronMulti.w.length; i++) {
			for (int l = 0; l < ImageOnlinePerceptronMulti.w[i].length; l++) {
				ImageOnlinePerceptronMulti.w[i][l] -= x[i]*eta*(y[l]-p[l]);
			}
		}
	}
	
	
	/*public static void main(String[] args) {
		for (int i = 0; i < x.length; i++) {
			if (GenRdm.nextBoolean()) {
				x[i] = 1;
				if (i < 10) y[i] = 1;
			} else {
				x[i] = 0;
				if (i < 10) y[i] = 0;
			}
		}
		w = InitialiseWeights();
		float[] test = InfPerceptron(w);
		System.out.println(Arrays.toString(test));
		float tot = 0;
		for (int i = 0; i < test.length; i++) {
			tot += test[i];
		}
		System.out.println(tot);
		
	}*/
}
