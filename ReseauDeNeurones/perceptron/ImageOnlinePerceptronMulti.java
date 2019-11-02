package perceptron;

import java.io.*;
import java.util.Arrays;

import mnisttools.MnistReader;

public class ImageOnlinePerceptronMulti {
    
	public static String path="/Users/Alan/Documents/JavaLibs//";
    public static String labelDB=path+"train-labels.idx1-ubyte";
    public static String imageDB=path+"train-images.idx3-ubyte";
    public static int DIM = 785;
    public final static int EPOCHMAX=40;
    public static float[][] w;
    
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
    *  1. on convertit l'image en deux dimension dx X dy, en un tableau unidimensionnel de tail dx.dy
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
    
    public static float[][] createImageSet(MnistReader db, int start, int nb) {
    	float[][] set = new float[nb][DIM];
    	for (int i = 0; i < nb; i++) {
    		set[i] = ConvertImage(BinariserImage(db.getImage(i+start), 100));
    	}
    	return set;
    }
    
    public static int[] createLabelSet(MnistReader db, int start, int nb) {
    	int[] set = new int[nb];
    	for (int i = 0; i < nb; i++) {
    		set[i] = db.getLabel(i + start);
    	}
    	return set;
    }
    
    public static int checkErr(int label, float[] probs) {
    	float max = 0;
    	int maxIdx = 0;
    	for (int i = 0; i < probs.length; i++) {
    		if (probs[i] > max) {
    			maxIdx = i;
    			max = probs[i];
    		}
    	}
    	if (maxIdx == label)
    		return 0;
    	return 1;
    }
    
    public static int epoch(float[][] wTemp, float x[][], int[] label, float eta) {
    	float[] y = new float[label.length];
    	int nbErr = 0;
    	int nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		y = perceptronMulti.InfPerceptron(w, x[i]);
    		perceptronMulti.updateWeights(x[i], y, perceptronMulti.OneHot(label[i]), eta);    		
    		nbErr += checkErr(label[i], y);
    	}
    	return nbErr;
    }
    
    public static int validation(float x[][], int[] label) {
    	float[] y = new float[label.length];
    	int nbErr = 0;
    	int nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		y = perceptronMulti.InfPerceptron(w, x[i]);
    		nbErr += checkErr(label[i], y);
    	}
    	return nbErr;
    }
    
    public static void main(String[] args) throws IOException {
		MnistReader db = new MnistReader(labelDB, imageDB);
		int Na = 5000;
		int Nv = 1000;
		System.out.print("Load database... ");
		float[][] trainData = createImageSet(db, 1, Na);
		int[] trainLabel = createLabelSet(db, 1, Na);
		float[][] validData = createImageSet(db, Na, Nv);
		int[] validLabel = createLabelSet(db, Na, Nv);
		w = perceptronMulti.InitialiseWeights(DIM);
		int[] nbErrT = new int[EPOCHMAX];
		int[] nbErrV = new int[EPOCHMAX];
		System.out.println("Done");
		System.out.print("Learning... ");
		File file = new File("data.d");
		file.createNewFile();
		FileWriter fw = new FileWriter(file);
		for (int i = 0; i < EPOCHMAX; i++) {
			nbErrT[i] = epoch(w, trainData, trainLabel, 0.001f);
			nbErrV[i] = validation(validData, validLabel);
			fw.write("" + i + " " + nbErrT[i]+ " " + nbErrV[i] + "\n");
		}
		fw.close();
		System.out.println("Done");
		System.out.println(Arrays.toString(nbErrT));
		System.out.println(Arrays.toString(nbErrV));
    }
}
