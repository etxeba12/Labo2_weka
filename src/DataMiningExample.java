
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IB1;
import weka.core.Capabilities;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class DataMiningExample {

    public static void main(String[] args) throws Exception {
        /////////////////////////////////////////////////////////////
        DataSource source = null;
        try {
            source = new DataSource("c:/Users/Iker/Desktop/Iker/UPV/4.curso/WEKA/Labos/Labo2/adult.train.arff");
        } catch (FileNotFoundException e) {
            System.out.println("ERROR: Revisar path del fichero de datos:"+args[0]);
        }

        Instances data=null;
        try {
            data = source.getDataSet();
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+args[0]);
        }

        data.setClassIndex(data.numAttributes()-1);


        /////////////////////////////////////////////////////////////
        AttributeSelection filter= new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search=new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        Instances newData = Filter.useFilter(data, filter);



        /////////////////////////////////////////////////////////////
        //
        NaiveBayes estimador= new NaiveBayes();

        //
        Evaluation evaluator = new Evaluation(newData);
        evaluator.crossValidateModel(estimador, newData, 10, new Random(1)); // Random(1): the seed=1 means "no shuffle" :-!
        //
        double acc=evaluator.pctCorrect();
        double inc=evaluator.pctIncorrect();
        double kappa=evaluator.kappa();
        double mae=evaluator.meanAbsoluteError();
        double rmse=evaluator.rootMeanSquaredError();
        double rae=evaluator.relativeAbsoluteError();
        double rrse=evaluator.rootRelativeSquaredError();
        double confMatrix[][]= evaluator.confusionMatrix();

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
