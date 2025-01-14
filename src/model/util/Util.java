package model.util;

import java.util.*;
import java.text.SimpleDateFormat;
import model.*;


public abstract interface Util {

    public static final int FACES = 19283;
    public static final int DIGITS = 12920202;

    public static final int FACE_IMAGE_LENGTH = 70;
    public static final int FACE_IMAGE_WIDTH = 60;

    public static final int DIGIT_IMAGE_LENGTH = 28;
    public static final int DIGIT_IMAGE_WIDTH = 28;

    public static final String DIGITS_HIDDEN_LAYER_WEIGHTS_DIR = "src/data/weights/digits/hidden/";
    public static final String DIGITS_OUTPUT_LAYER_WEIGHTS_DIR = "src/data/weights/digits/output/";

    public static final String TRAINING_IMAGES_DATA = "src/model/reader/data/train-images.idx3-ubyte";
    public static final String TRAINING_IMAGES_LABELS = "src/model/reader/data/train-labels.idx1-ubyte";

    public static final String TEST_IMAGES_DATA = "src/model/reader/data/t10k-images.idx3-ubyte";
    public static final String TEST_IMAGES_LABELS = "src/model/reader/data/t10k-labels.idx1-ubyte";


    public static final String DIGIT_OUTPUT_TRAINING_DIRECTORY = "src/data/output/digits/training/";
    public static final String DIGIT_OUTPUT_DEMO_DIRECTORY = "src/data/output/digits/demo/";

    public static final String FACES_OUTPUT_TRAINING_DIRECTORY = "src/data/output/faces/training/";
    public static final String FACES_OUTPUT_DEMO_DIRECTORY = "src/data/output/faces/demo/";

    public static final String DIGIT_TRAINING_DATA = "src/data/digitdata/trainingimages";
    public static final String DIGIT_TEST_DATA = "src/data/digitdata/testimages";
    public static final String DIGIT_VALIDATION_DATA = "src/data/digitdata/validationimages";

    public static final String DIGIT_TRAINING_LABELS = "src/data/digitdata/traininglabels";
    public static final String DIGIT_TEST_LABELS = "src/data/digitdata/testlabels";
    public static final String DIGIT_VALIDATION_LABELS = "src/data/digitdata/validationlabels";

    public static final String FACE_TRAINING_DATA = "src/data/facedata/facedatatrain";
    public static final String FACE_VALIDATION_DATA = "src/data/facedata/facedatavalidation";
    public static final String FACE_TEST_DATA = "src/data/facedata/facedatatest";

    public static final String FACE_TRAINING_LABELS = "src/data/facedata/facedatatrainlabels";
    public static final String FACE_VALIDATION_LABELS = "src/data/facedata/facedatavalidationlabels";
    public static final String FACE_TEST_LABELS = "src/data/facedata/facedatatestlabels";

    public static final String FACE_WEIGHTS_DIRECTORY = "src/data/weights/faces/";



    public static final long seed = 1;
    public static final Random random = new Random();


    public static double sigmoid(double x) {
        double e = Math.exp(-x);
        return 1/(1+e);
    }

    public static String hhmmss(long milliseconds) {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
        sdf.setTimeZone(TimeZone.getTimeZone("GMT")); // Setting timezone to GMT to ensure accurate calculation
        String formattedTime = sdf.format(new Date(milliseconds));
        return formattedTime;
    }

    public static double[] toDoubleArray(int[] arr) {
        double[] out = new double[arr.length];
        for(int i = 0; i< arr.length; i++) {
            out[i] = (double)(arr[i]);
        }
        return out;
    }

    public static List<Image> copy(List<Image> list) {
        List<Image> copy = new ArrayList<>();
        for(Image i : list) {
            copy.add(i);
        }
        return copy;
    }
}
