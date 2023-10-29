package ml.classifiers;

import java.util.Random;

import ml.Example;
import ml.DataSet;

/**
 * A classifier that randomly labels examples as either -1 or 1.
 * 
 * @author dkauchak
 *
 */
public class RandomClassifier implements Classifier {
    private Random rand = new Random();

    @Override
    public void train(DataSet data) {
        // easiest training method ever!
    }

    @Override
    public double classify(Example example) {
        return rand.nextInt(2) == 0 ? 1 : -1;
    }
}