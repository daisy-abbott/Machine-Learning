package ml.classifiers;

import ml.Example;
import ml.DataSet;
import java.util.Set;
import java.sql.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Class that implements the classifier interface and predicts the label based
 * on the learned
 * tree.
 * 
 * @author Daisy Abbott + Vivien Song
 *         added split feature method, moved return statement in training error
 */

// implement classifier interface
public class DecisionTreeClassifier implements Classifier {
    DecisionTreeNode root;
    DataSet dataset = new DataSet("/Users/viviensong/cs158/assignment-2-daisy/data/titanic-train.csv"); // DataSet
                                                                                                        // object
    double savedIndex;
    ArrayList<Example> allExamples = dataset.getData();
    public int depthLimit;
    public int stopDepth;
    ArrayList<Example> listdata = new ArrayList<Example>();

    /*
     * Constructor for DTClassifier
     */
    public DecisionTreeClassifier() {
    }

    @Override // method from interface
    /*
     * This method trains the DecisionTree on the dataset by calling buildTree from
     * a root
     * 
     * @param DataSet: data
     * 
     * @returns void
     */
    public void train(DataSet data) {
        listdata = data.getData();
        this.depthLimit = depthLimit;
        root = buildTree(listdata, depthLimit, new HashSet<Double>());
        this.root = root;
        this.dataset = dataset;
    }

    /*
     * This method recursively builds the tree by adding nodes for each label to
     * split on
     * and only stops when it hits one of five base cases.
     * 
     * @args ArrayList<Example>: listData, int: depth, Set<Double>: usedFeatures
     * 
     * @returns DecisionTreeNode
     */

    public DecisionTreeNode buildTree(ArrayList<Example> listdata, int depth, Set<Double> usedFeatures) {
        this.listdata = listdata;

        // First base case: if all data is same class label, return leaf node with that
        // label
        if (IsAllSameClass(listdata)) {
            double same = listdata.get(0).getLabel();
            DecisionTreeNode label = new DecisionTreeNode(same);
            return label;

        }

        // Third base case: if we have looked at all the features, return majority label
        // of parent
        if (usedFeatures.size() == listdata.get(0).getFeatureSet().size()) {
            return new DecisionTreeNode(MajorityLabel(listdata));
        }

        // Fifth base case: if we have gone over the depth limit, return the majority
        // label of parent
        if (depthLimit >= stopDepth) {
            return new DecisionTreeNode(MajorityLabel(listdata));
        }

        // Grabs index of feature with lowest training error
        int lowestError = trainingError(listdata, usedFeatures);

        // Second base case: if all features have the same value, return majority label
        if (IsAllSameValue(listdata)) {
            return new DecisionTreeNode(MajorityLabel(listdata));
        }

        // Recursively split data:
        // 1. partition data into data_left and data_right
        ArrayList<Example> dataRight = splitRight(listdata, lowestError);
        ArrayList<Example> dataLeft = splitLeft(listdata, lowestError);

        // 2. Keep track of used features by making copy of features in set
        Double converted = Double.valueOf(lowestError);
        Set<Double> usedFeaturesCopy = new HashSet<Double>(usedFeatures);
        usedFeaturesCopy.add(converted); // add used features to set

        // 3. Create new parent node from lowest error
        DecisionTreeNode parentNode = new DecisionTreeNode(lowestError);

        this.depthLimit = depthLimit + 1;

        // 4. Recurse on left and right after checking that the dataset is not empty and
        // are not all the same values
        if (dataRight.isEmpty() || IsAllSameValue(dataRight)) {
            parentNode.setRight(new DecisionTreeNode(MajorityLabel(listdata))); // Return the majority label of the
                                                                                // parent node ???!??
        } else {
            parentNode.setRight(buildTree(dataRight, (depthLimit++), usedFeaturesCopy));

        }

        if (dataLeft.isEmpty() || IsAllSameValue(dataLeft)) {
            parentNode.setLeft(new DecisionTreeNode(MajorityLabel(listdata))); // Return the majority label of the
                                                                               // parent node ???!??
        } else {
            parentNode.setLeft(buildTree(dataLeft, (depthLimit++), usedFeaturesCopy));

        }
        // Return tree
        return parentNode;
    }

    /*
     * This method checks if all values are of the same class label. Picks that
     * label if so
     * 
     * @param ArrayList<Example>: arrayData
     * 
     * @returns boolean
     */
    public boolean IsAllSameClass(ArrayList<Example> arrayData) {
        // keep track of if same value
        double same = 0.0;
        for (int i = 0; i < arrayData.size(); i++) {
            if (i == 0) { // on first feature, save value.
                same = arrayData.get(i).getLabel();
            } else {
                if (arrayData.get(i).getLabel() == same) { // for all other values are same, return true
                    continue;
                } else {
                    return false;
                }
            }

        }
        return true;

    }

    /*
     * This method checks if all the data have the same feature values and if so
     * picks majority label
     * 
     * @params ArrayList<Example>: arrayData
     * 
     * @return boolean
     */
    public boolean IsAllSameValue(ArrayList<Example> arrayData) {
        Example tester;
        tester = arrayData.get(0);// grabs first row
        for (int i = 0; i < arrayData.size(); i++) {
            if (tester.equalFeatures(arrayData.get(i)) == true) {
                continue;
            } else {
                return false;
            }

        }
        return true;
    }

    /*
     * This method returns a double with the majority label of the feature
     * 
     * @params ArrayList<Example>: arrayData
     * 
     * @returns double
     */
    public double MajorityLabel(ArrayList<Example> arrayData) {
        ArrayList<Double> classLabel = new ArrayList<Double>();
        int count1 = 0;
        int count2 = 0;
        // loop through dataset, add label to array list

        for (int i = 0; i < arrayData.size(); i++) {
            classLabel.add(arrayData.get(i).getLabel());

        }
        // loop through list of class labels, divide into bins for counting
        for (int i = 0; i < classLabel.size(); i++) {
            double current = classLabel.get(i);
            if (current == 1) {
                count1++;
            } else {
                count2++;
            }
        }
        double majority = 0;

        if (count1 > count2) {
            majority = 1.0;
        } else if (count1 == count2) {
            majority = classLabel.get(0);
        } else {
            majority = -1.0;
        }

        return majority;
    }

    /*
     * This method splits the data into a dataRight arrayList containing only
     * examples with a feature
     * value of 1.
     * 
     * @args ArrayList<Example>: arrayData, int: feature
     * 
     * @returns ArrayList<Example>
     */
    public ArrayList<Example> splitRight(ArrayList<Example> arrayData, int feature) {
        // DataSet[] split = new DataSet [2];
        ArrayList<Example> split = new ArrayList<Example>();
        for (int i = 0; i < arrayData.size(); i++) {
            if (arrayData.get(i).getFeature(feature) == 1) { // if row containing given feature is 0
                split.add(arrayData.get(i));
            }
        }
        return split;
    }

    /*
     * This method splits the data into a dataLeft arrayList containing only
     * examples with a feature
     * value of 0.
     * 
     * @args ArrayList<Example>: arrayData, int: feature
     * 
     * @returns ArrayList<Example>
     */
    public ArrayList<Example> splitLeft(ArrayList<Example> arrayData, int feature) {
        ArrayList<Example> split = new ArrayList<Example>();
        for (int i = 0; i < arrayData.size(); i++) {
            if (arrayData.get(i).getFeature(feature) == 0) { // if row containing given feature is 0
                split.add(arrayData.get(i));
            }
        }
        return split;
    }

    /*
     * This method calculates the training errors of different unused features and
     * returns
     * the index of the feature with the lowest training error.
     * 
     * @params ArrayList<Example>: arrayData, Set<Double>: usedFeatures
     * 
     * @returns int
     */
    public int trainingError(ArrayList<Example> arrayData, Set<Double> usedFeatures) {

        List<Integer> featureSet = new ArrayList<>(arrayData.get(0).getFeatureSet());
        float min = Float.MAX_VALUE;
        for (Integer feature : featureSet) {
            Double featureConverted = Double.valueOf(feature);
            if (!usedFeatures.contains(featureConverted)) {
                ArrayList<Double> trackZero = new ArrayList<Double>();
                ArrayList<Double> trackOne = new ArrayList<Double>();
                int survivedZero = 0;
                int survivedOne = 0;
                int deadZero = 0;
                int deadOne = 0;
                for (int j = 0; j < arrayData.size(); j++) { // loops through all examples in dataset
                    if (arrayData.get(j).getFeature(feature) == 0.0) {
                        trackZero.add(arrayData.get(j).getLabel()); // adding all the survival vaklues into array
                    } else {

                        trackOne.add(arrayData.get(j).getLabel());
                    }
                }

                for (Double survival : trackZero) {
                    int converted = survival.intValue();
                    if (converted == 1) {
                        survivedZero++;
                    } else {
                        deadZero++;
                    }
                }

                for (Double survival : trackOne) {
                    int converted = survival.intValue();
                    if (converted == 1) {
                        survivedOne++;
                    } else {
                        deadOne++;
                    }
                }
                int majorityZero = Math.max(deadZero, survivedZero);
                int majorityOne = Math.max(deadOne, survivedOne);
                int total = allExamples.size();
                float accuracy = (majorityZero + majorityOne);
                accuracy = accuracy / total;
                float error = 1 - accuracy;

                // code to save lowest error
                if (min > error) {
                    min = error;
                    savedIndex = feature; // save index of feature
                }

            }
        }
        int converted = (int) savedIndex;
        return converted; // returns the converted int (converted from double)
    }

    @Override // from interface

    /*
     * Calls traverseClassify to return a double with the prediction for the example
     * 
     * @params Example: example
     * 
     * @returns double
     */
    public double classify(Example example) {
        return traverseClassify(root, example);
    }

    /*
     * This method returns the prediction from a leaf node through recursive calls
     * 
     * @params DecisionTreeNode: node, Example: example
     * 
     * @returns double
     */
    private double traverseClassify(DecisionTreeNode node, Example example) {
        if (node.isLeaf()) {
            return node.prediction();
        }

        // Check if feature is present in the example
        double feature = example.getFeature(node.getFeatureIndex());

        // Descend tree based on presence of the feature
        if (feature == 0.0) {
            return traverseClassify(node.getLeft(), example);
        } else if (feature == 1.0) {
            return traverseClassify(node.getRight(), example);
        } else {
            return -1;
        }
    }

    /*
     * This method sets the maximum depth limit of the tree to allow for a premature
     * stop
     * 
     * @params int: depth
     * 
     * @returns void
     */
    public void setDepthLimit(int depth) {
        this.stopDepth = depth;
    }

    /*
     * This method prints the entire tree
     */
    public String toString() {
        // Call the treeString method of the root node to print the entire tree
        if (root != null) {
            return root.treeString(dataset.getFeatureMap());
        } else {
            return "Decision tree is not trained yet.";
        }
    }

    public static void main(String[] args) {
        DecisionTreeClassifier test = new DecisionTreeClassifier();
        DataSet dataset = new DataSet("/Users/viviensong/cs158/assignment-2-daisy/data/titanic-train.csv"); // DataSet
                                                                                                            // object
        test.train(dataset);
    }
}