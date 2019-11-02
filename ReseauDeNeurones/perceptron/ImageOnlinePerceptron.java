package perceptron;

import java.io.*;

import java.util.Arrays;
import java.util.Random;
import mnisttools.MnistReader;

public class ImageOnlinePerceptron {

    /* Les donnees */
    public static String path="/Users/Alan/Documents/JavaLibs//";
    public static String labelDB=path+"train-labels.idx1-ubyte";
    public static String imageDB=path+"train-images.idx3-ubyte";

    /* Parametres */
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 1000; 
    // Nv exemples pour l'ensemble d'évaluation
    public static final int Nv = 1000; 
    // Nombre d'epoque max
    public final static int EPOCHMAX=40;
    // Classe positive (le reste sera considere comme des ex. negatifs):
    public static int  classe = 6 ;
    public static int DIM = 785;
    public static float[] w = new float[DIM];

    // Générateur de nombres aléatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random();//(seed);

    /*
    *  BinariserImage : 
    *      image: une image int à deux dimensions (extraite de MNIST)
    *      seuil: parametre pour la binarisation
    *
    *  on binarise l'image à l'aide du seuil indiqué
    *
    */
    public static int[][] BinariserImage(int[][] image, int seuil) {
    	int[][] binaryImage = new int[image.length][image[0].length];
    	for (int i = 0; i < image.length; i++) {
    		for (int j = 0; j < image[i].length; j++) {
    			if (image[i][j] >= seuil) {
    				binaryImage[i][j] = 1;
    			} else {
    				binaryImage[i][j] = 0;
    			}
    		}
    	}
    	return binaryImage;
    }

    /*
    *  ConvertImage : 
    *      image: une image int binarisée à deux dimensions
    *
    *  1. on convertit l'image en deux dimension dx X dy, en un tableau unidimensionnel de taille dx.dy
    *  2. on rajoute un élément en première position du tableau qui sera à 1
    *  La taille finale renvoyée sera dx.dy + 1
    *
    */
    public static float[] ConvertImage(int[][] image) {
        float[] convertedImage = new float[DIM];
        convertedImage[0] = 1;
        for (int i = 0; i < image.length; i++) {
        	for (int j = 0; j < image[i].length; j++) {
        		convertedImage[i*image[i].length + j + 1] = image[i][j];
        	}
        }
        return convertedImage;
    }

    /*
    *  InitialiseW :
    *      sizeW : la taille du vecteur de poids
    *      alpha : facteur à rajouter devant le nombre aléatoire
    *
    *  le vecteur de poids est crée et initialisé à l'aide d'un générateur
    *  de nombres aléatoires.
    */
    public static float[] InitialiseW(int sizeW, float alpha) {
    	float[] w = new float[sizeW];
    	for (int i = 0; i < sizeW; i++) {
    		w[i] = alpha*(GenRdm.nextFloat()-0.5f);
    	}
    	return w;
    }
    
    public static float dot(float[] x, float[] y) {
    	float somme = 0;
    	for (int  i = 0; i < DIM; i++) {
    		somme += x[i]*y[i];
    	}
    	return somme;
    }
    
    public static void majParams(float [] w_ancien, float[] x, int y, float eta) {
    	for (int i = 0; i < DIM; i++) {
    		w[i] = w_ancien[i] + eta*y*x[i];
    	}
    }

    public static int prediction(float[] w, float[] x) {
    	float yPrediction = dot(x, w);
    	if (yPrediction >= 0) {
    		return 1;
    	} else {
    		return -1;
    	}
    }
    
    public static int epoch(float[] wTemp, float[][] x, int[] y, float eta) {
    	int nbErr = 0;
    	int nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		float[] xCourant = x[i];
    		int yCourant = y[i];
    		int yPrediction = prediction(wTemp, xCourant);
    		if (yCourant != yPrediction) {
    			majParams(wTemp, xCourant, yCourant, eta);
    			nbErr++;
    		}
    	}
    	return nbErr;
    }
    
    public static int countError(float[][] x, int[] y) {
    	int nbErr = 0;
    	int nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		float[] xCourant = x[i];
    		int yCourant = y[i];
    		int yPrediction = prediction(w, xCourant);
    		if (yCourant != yPrediction) {
    			nbErr++;
    		}
    	}
    	return nbErr;
    }

    public static void main(String[] args) {
        System.out.println("# Load the database !");
        /* Lecteur d'image */ 
        MnistReader db = new MnistReader(labelDB, imageDB);
        float[][] trainData = new float[Na][DIM];
        int[] trainRef = new int[Na];
        for (int i = 1; i < Na; i++) {
        	int[][] binImage = BinariserImage(db.getImage(i), 150);
        	trainData[i] = ConvertImage(binImage);
        	int label = db.getLabel(i);
        	if (label == classe) {
        		trainRef[i] = 1;
        	} else {
        		trainRef[i] = -1;
        	}
        }
        float[][] validData = new float[Nv][DIM];
        int[] validRef = new int[Nv];
        for (int i = 0; i < Nv; i++) {
        	int[][] binImage = BinariserImage(db.getImage(i+40000), 150);
        	validData[i] = ConvertImage(binImage);
        	int label = db.getLabel(i+40000);
        	if (label == classe) {
        		validRef[i] = 1;
        	} else {
        		validRef[i] = -1;
        	}
        }
        /* Creation des donnees */
        System.out.println("# Build train for digit "+ classe);
        /* Tableau où stocker les données */
        w = InitialiseW(DIM, 1);
        int[] nbErr = new int[EPOCHMAX];
        int[] nbErrValid = new int[EPOCHMAX];
        for (int i = 0; i < EPOCHMAX; i++) {
        	nbErr[i] = epoch(w, trainData, trainRef, 0.2f);
        	nbErrValid[i] = countError(validData, validRef);
        }
        System.out.println(Arrays.toString(nbErr));
        System.out.println(Arrays.toString(nbErrValid));
    }
}
