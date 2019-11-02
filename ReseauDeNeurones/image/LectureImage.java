package image;
import mnisttools.MnistReader;

public class LectureImage {

	public static void main(String[] args) {
		String path = "/Users/Alan/Documents/JavaLibs/";
		String labelDB = path + "train-labels.idx1-ubyte"; 
		String imageDB = path + "train-images.idx3-ubyte";
		// Creation de la base de donnees
		MnistReader db = new MnistReader(labelDB, imageDB);
		// Acces a la premiere image
		int idx = 1; // Attention premiere valeur est 1
		int[][] image = db.getImage(idx);
		int label = db.getLabel(idx);
		// Affichage du label
		System.out.print("Le label est " + label + "\n");
		// Affichage du nombre total d'images
		System.out.print("Le nombre d'images est " + db.getTotalImages() + "\n");
		// Affichage de la taille de l'image
		System.out.print("La taille de l'image est " + image.length + "x" + image[0].length + "\n");
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[i].length; j++) {
				System.out.print(image[i][j] + " ");
			}
			System.out.print("\n");
		}
	}

}
