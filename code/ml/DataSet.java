package ml;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * A collections of examples representing an entire data set.
 * 
 * @author dkauchak
 */
public class DataSet {
    private String[] headers; // the headers/feature names
    private ArrayList<Example> data = new ArrayList<Example>(); // the data/examples in this data set
    // the mapping from feature indices to the name of the feature
    private HashMap<Integer, String> featureMap = new HashMap<Integer, String>();

    /**
     * Creates a new data set from a CSV file. The file can start with any number
     * of "comment" lines which must start with a # sound. Then the next line must
     * be a header (i.e. the features) then all following lines are treated as
     * examples.
     * 
     * Assumes the label is the last column.
     * 
     * @param csvFile comma separated file containing the examples WITH a header
     */
    public DataSet(String csvFile) {
        int numColumns = -1;

        // figure out how many columns there are then call
        try {
            BufferedReader in = new BufferedReader(new FileReader(csvFile));

            // ignore any lines at the beginning that start with #
            String line = in.readLine();

            while (line.startsWith("#")) {
                line = in.readLine();
            }

            // parse the headers
            numColumns = line.split(",").length - 1;
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        initialize(csvFile, numColumns);
    }

    /**
     * Creates a new data set from a CSV file. The file can start with any number
     * of "comment" lines which must start with a # sound. Then the next line must
     * be a header (i.e. the features) then all following lines are treated as
     * examples.
     * 
     * @param csvFile    comma separated file containing the examples WITH a header
     * @param labelIndex the index (0-based) where the label is at
     */
    public DataSet(String csvFile, int labelIndex) {
        initialize(csvFile, labelIndex);
    }

    private void initialize(String csvFile, int labelIndex) {
        try {
            BufferedReader in = new BufferedReader(new FileReader(csvFile));

            // ignore any lines at the beginning that start with #
            String line = in.readLine();

            while (line.startsWith("#")) {
                line = in.readLine();
            }

            // parse the headers
            headers = line.split(",");

            int featureIndex = 0;

            for (int i = 0; i < headers.length; i++) {
                if (i != labelIndex) {
                    featureMap.put(featureIndex, headers[i]);
                    featureIndex++;
                }
            }

            CSVDataReader reader = new CSVDataReader(in, labelIndex);

            while (reader.hasNext()) {
                Example next = reader.next();
                data.add(next);
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Internal constructor for copying the data set (or most of it)
     * 
     * @param s
     */
    @SuppressWarnings("unchecked")
    private DataSet(DataSet s) {
        this.headers = s.headers.clone();
        this.featureMap = (HashMap<Integer, String>) s.featureMap.clone();
    }

    /**
     * Get the mapping from feature indices to feature names. This is
     * mostly useful when trying to print out the final models.
     * 
     * @return feature map
     */
    public HashMap<Integer, String> getFeatureMap() {
        return featureMap;
    }

    /**
     * Get the examples associated with this data set
     * 
     * @return the examples
     */
    public ArrayList<Example> getData() {
        return data;
    }

    /**
     * Get all of the feature indices that are contained in this
     * data set.
     * 
     * @return
     */
    public Set<Integer> getAllFeatureIndices() {
        return featureMap.keySet();
    }

    /**
     * Split this data set into two data sets of size:
     * - total_size * fraction
     * - total_size - (total_size*fraction)
     * 
     * @param fraction the proportion to allocate to the first data set in the split
     * @return two data sets representing a fraction split of the data
     */
    @SuppressWarnings("unchecked")
    public DataSet[] split(double fraction) {
        ArrayList<Example> newdata = (ArrayList<Example>) data.clone();
        Collections.shuffle(newdata, new Random(System.nanoTime()));

        ArrayList<Example> train = new ArrayList<Example>();
        ArrayList<Example> test = new ArrayList<Example>();

        int trainSize = (int) Math.floor(data.size() * fraction);

        for (int i = 0; i < newdata.size(); i++) {
            if (i < trainSize) {
                train.add(newdata.get(i));
            } else {
                test.add(newdata.get(i));
            }
        }

        DataSet[] splits = new DataSet[2];
        splits[0] = new DataSet(this);
        splits[0].data = train;

        splits[1] = new DataSet(this);
        splits[1].data = test;

        return splits;
    }
}