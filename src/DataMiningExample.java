
import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IB1;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;


///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class DataMiningExample {

    public static void main(String[] args) throws Exception {
        /////////////////////////////////////////////////////////////
        String path1 = "c:/Users/Iker/Desktop/Iker/UPV/4.curso/WEKA/Labos/Labo2/breast-cancer.arff";
        String path2  = "c:/Users/Iker/Desktop/Iker/UPV/4.curso/WEKA/Labos/Labo2/prueba.arff";
        DataSource source = null;
        try {
            source = new DataSource(path1);
        } catch (FileNotFoundException e) {
            System.out.println("ERROR: Revisar path del fichero de datos:"+args[0]);
        }

        Instances data=null;
        try {
            data = source.getDataSet();
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+args[0]);
        }


        Randomize filter = new Randomize();
        filter.setInputFormat(data);
        Instances randomData = Filter.useFilter(data,filter);

        RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(randomData);
        filterRemove.setPercentage(34); //quita de randomDate el 34% de las instancias empezando desde arriba

        Instances train = Filter.useFilter(randomData,filterRemove); //el 100% menos el 34% es lo que se queda en el train
        System.out.println("Trainen istantzia kopurua: " + train.numInstances());

        RemovePercentage filterRemove2 = new RemovePercentage();
        filterRemove2.setInputFormat(randomData);
        filterRemove2.setPercentage(34);
        filterRemove2.setInvertSelection(true); //quita el 66% de las instancias empezando desde abajo

        Instances test = Filter.useFilter(randomData, filterRemove2);
        test.setClassIndex(test.numAttributes()-1);


        System.out.println("Test-en istantzia kopurua: " + test.numInstances());
        train.setClassIndex(train.numAttributes()-1);

        NaiveBayes clasifier = new NaiveBayes();
        clasifier.buildClassifier(train);

        Evaluation evaluator = new Evaluation(train);
        evaluator.evaluateModel(clasifier,test);



        ///////////////////////////////////////////
        double acc=evaluator.pctCorrect();
        double inc=evaluator.pctIncorrect();
        double kappa=evaluator.kappa();
        double mae=evaluator.meanAbsoluteError();
        double rmse=evaluator.rootMeanSquaredError();
        double rae=evaluator.relativeAbsoluteError();
        double rrse=evaluator.rootRelativeSquaredError();
        double confMatrix[][]= evaluator.confusionMatrix();

        BufferedWriter writer = new BufferedWriter(new FileWriter(path2));
        Date date = new Date();
        writer.write("Exekuzio data: " + date);
        writer.newLine();

        writer.write(path1);
        writer.newLine();

        writer.write(path2);
        writer.newLine();

        writer.write(evaluator.toMatrixString());
        writer.write("\nEbaluazio metrikak\n========================\nKlase minoritarioarena: \nAccuracy "+evaluator.pctCorrect());
        writer.write("\nWeighted Avg.: \nPrecision: "+ evaluator.weightedPrecision() + "\nRecall "+ evaluator.weightedRecall() +" \nF-measure "+evaluator.weightedFMeasure());
        writer.write(evaluator.toSummaryString("\nResults\n========================\n", false));
        writer.write("Estimated Accuracy: " + Double.toString(evaluator.pctCorrect()) + "\n");
        writer.write(evaluator.toClassDetailsString());


        writer.flush();
        writer.close();

        System.out.println("Correctly Classified Instances  " + acc);
        System.out.println("Incorrectly Classified Instances  " + inc);
        System.out.println("Kappa statistic  " + kappa);
        System.out.println("Mean absolute error  " + mae);
        System.out.println("Root mean squared error  " + rmse);
        System.out.println("Relative absolute error  " + rae);
        System.out.println("Root relative squared error  " + rrse);

        System.out.println();
        System.out.println();

        int i = 0;
        int a = 0;
        String lista[] = {"a","b","c","d","e"};
        System.out.println("=== Confusion Matrix ===");
        System.out.println("     a     b   <-- Classified as");
        while (i < confMatrix.length){
            while(a < confMatrix.length){
                System.out.print(confMatrix[i][a] + " ");
                a ++;
            }
            System.out.print("|    " + lista[i] + " = " + data.attribute(data.numAttributes()-1).value(i));
            System.out.println();
            a = 0;
            i ++;

        }
    }
}
