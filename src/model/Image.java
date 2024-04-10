package model;

import java.io.*;

public class Image extends AbstractImage {

    public Image(int n, int a, int b, RandomAccessFile file, int type) {
        this.type = type;
        this.n = n;
        this.a = a;
        this.b = b;
        image = new int[DIGIT_IMAGE_LENGTH][DIGIT_IMAGE_WIDTH];
        try {
            if(type == FACES) {
                file.readLine();
            }
            for(int i = 0; i < image.length; i++) {
                for(int j = 0; j < image[i].length; j++) {
                    int c = file.read();
                    if(c != -1) {
                        image[i][j] = (char)(c);
                    }
                }
            }
            file.readLine();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        calc_phi(n, a, b);
    }

    public Image(int n, int a, int b, int[][] matrix) {
        this.image = matrix;
        this.n = n;
        this.a = a;
        this.b = b;
        calc_phi(n, a, b);
    }

    protected boolean validDimensions(int n, int a, int b) {
        if(n * a * b != 28 * 28) {
            throw new IllegalArgumentException("n*a*b must equal 784");
        }
        if(28%a != 0) {
            throw new IllegalArgumentException("a must divide 28");
        }
        if(28%b != 0) {
            throw  new IllegalArgumentException("b must divide 28");
        }
        return true;
    }
}
