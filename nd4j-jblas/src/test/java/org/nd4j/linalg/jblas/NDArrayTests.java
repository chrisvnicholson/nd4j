package org.nd4j.linalg.jblas;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.util.ArrayUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests extends org.nd4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);


    @Test
    public void testMatrixVector() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };


        NDArrays.factory().setOrder('f');
        double[] mmul = {1,2,3,4};

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = NDArrays.create(data);
        assertEquals(d.rows,d2.rows());
        assertEquals(d.columns,d2.columns());
        verifyElements(d,d2);

        INDArray toMmulD2 = NDArrays.create(mmul).transpose();
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);


        assertEquals(d.rows,d2.rows());
        assertEquals(d.columns,d2.columns());

        assertEquals(toMmulD.rows,toMmulD2.rows());
        assertEquals(toMmulD.columns,toMmulD2.columns());

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);

        verifyElements(mmulResultD,mmulResultD2);





        NDArrays.factory().setOrder('c');


    }

    @Test
    public void testTransposeCompat() {
        NDArrays.factory().setOrder('f');
        DoubleMatrix dReshaped = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray nReshaped = NDArrays.linspace(1,8,8).reshape(2,4);
        verifyElements(dReshaped,nReshaped);
        DoubleMatrix d = dReshaped.transpose();
        INDArray n = nReshaped.transpose();
        verifyElements(d,n);
        assertTrue(ArrayUtil.equals(n.data(),d.data));
        NDArrays.factory().setOrder('c');

    }


    @Test
    public void testFortranRavel() {
        double[][] data = new double[][] {
                {1,2,3,4},
                {5,6,7,8}
        };

        INDArray toRavel = NDArrays.create(data);
        NDArrays.factory().setOrder('f');
        INDArray toRavelF = NDArrays.create(data);
        INDArray ravel = toRavel.ravel();
        INDArray ravelF = toRavelF.ravel();
        assertEquals(ravel,ravelF);
        NDArrays.factory().setOrder('c');

    }


    @Test
    public void testNorm1() {
        DoubleMatrix norm1 = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray norm1NDArray = NDArrays.linspace(1,8,8).reshape(2,4);
        assertEquals(norm1.norm1(),norm1NDArray.norm1(Integer.MAX_VALUE).get(0),1e-1);
    }



    @Test
    public void testFortranReshapeMatrix() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };

        NDArrays.factory().setOrder('f');

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = NDArrays.create(data);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);


        DoubleMatrix reshapedD = d.reshape(4,2);
        INDArray reshapedD2 = d2.reshape(4,2);
        verifyElements(reshapedD,reshapedD2);
        NDArrays.factory().setOrder('c');


    }






    @Test
    public void testFortranCreation() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };


        NDArrays.factory().setOrder('f');
        float[][] mmul = {{1,2,3,4},{5,6,7,8}};

        INDArray d2 = NDArrays.create(data);
        verifyElements(mmul,d2);
    }


    @Test
    public void testMatrixMatrix() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };


        NDArrays.factory().setOrder('f');
        double[][] mmul = {{1, 2, 3, 4}, {5, 6, 7, 8}};

        DoubleMatrix d = new DoubleMatrix(data).reshape(4, 2);
        INDArray d2 = NDArrays.create(data).reshape(4, 2);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);

        INDArray toMmulD2 = NDArrays.create(mmul);
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);
        verifyElements(mmulResultD, mmulResultD2);


        NDArrays.factory().setOrder('c');
    }

    @Test
    public void testVectorVector() {
        DoubleMatrix d = new DoubleMatrix(2,1);
        d.data = new double[]{1,2};
        DoubleMatrix d2 = new DoubleMatrix(1,2);
        d2.data = new double[]{3,4};

        INDArray d3 = NDArrays.create(new double[]{1,2}).reshape(2,1);
        INDArray d4 = NDArrays.create(new double[]{3,4});

        assertEquals(d.rows,d3.rows());
        assertEquals(d.columns,d3.columns());

        assertEquals(d2.rows,d4.rows());
        assertEquals(d2.columns,d4.columns());

        DoubleMatrix resultMatrix = d.mmul(d2);



        INDArray resultNDArray = d3.mmul(d4);
        verifyElements(resultMatrix,resultNDArray);

    }


    @Test
    public void testVector() {
        NDArrays.factory().setOrder('f');

        DoubleMatrix dJblas = DoubleMatrix.linspace(1,4,4);
        INDArray d = NDArrays.linspace(1,4,4);
        verifyElements(dJblas,d);
        NDArrays.factory().setOrder('c');


    }
    @Test
    public void testRowVectorOps() {
        if(NDArrays.factory().order() ==  NDArrayFactory.C) {
            INDArray twoByTwo = NDArrays.create(new float[]{1,3,2,4},new int[]{2,2});
            INDArray toAdd = NDArrays.create(new float[]{1,2},new int[]{2});
            twoByTwo.addiRowVector(toAdd);
            INDArray assertion = NDArrays.create(new float[]{2,3,5,6},new int[]{2,2});
            assertEquals(assertion,twoByTwo);

        }



    }

    @Test
    public void testColumnVectorOps() {
        if(NDArrays.factory().order() == NDArrayFactory.C) {
            INDArray twoByTwo = NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
            INDArray toAdd = NDArrays.create(new float[]{1,2},new int[]{2,1});
            twoByTwo.addiColumnVector(toAdd);
            INDArray assertion = NDArrays.create(new float[]{2,3,5,6},new int[]{2,2});
            assertEquals(assertion,twoByTwo);


        }


    }

    @Test
    public void testReshapeCompatibility() {
        NDArrays.factory().setOrder('f');
        DoubleMatrix oneThroughFourJblas = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        DoubleMatrix fiveThroughEightJblas = DoubleMatrix.linspace(5,8,4).reshape(2,2);
        INDArray oneThroughFour = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray fiveThroughEight = NDArrays.linspace(5,8,4).reshape(2,2);
        verifyElements(oneThroughFourJblas,oneThroughFour);
        verifyElements(fiveThroughEightJblas,fiveThroughEight);
        NDArrays.factory().setOrder('c');

    }

    @Test
    public void testRowSumCompat() {
        NDArrays.factory().setOrder('f');
        DoubleMatrix rowsJblas = DoubleMatrix.linspace(1,8,8).reshape(2,4);
        INDArray rows = NDArrays.linspace(1,8,8).reshape(2,4);
        verifyElements(rowsJblas,rows);

        INDArray rowSums = rows.sum(1);
        DoubleMatrix jblasRowSums = rowsJblas.rowSums();
        verifyElements(jblasRowSums,rowSums);


        float[][] data = new float[][]{
                {1,2},{3,4}
        };

        INDArray rowSumsData = NDArrays.create(data);
        NDArrays.factory().setOrder('c');
        INDArray rowSumsCOrder = NDArrays.create(data);
        assertEquals(rowSumsData,rowSumsCOrder);
        INDArray rowSumsDataSum = rowSumsData.sum(1);
        INDArray rowSumsCOrderSum = rowSumsCOrder.sum(1);
        assertEquals(rowSumsDataSum,rowSumsCOrderSum);
        INDArray assertion = NDArrays.create(new float[]{3,7});
        assertEquals(assertion,rowSumsCOrderSum);
        assertEquals(assertion,rowSumsDataSum);
    }



    protected void verifyElements(float[][] d,INDArray d2) {
        for(int i = 0; i < d2.rows(); i++) {
            for(int j = 0; j < d2.columns(); j++) {
                float test1 =  d[i][j];
                float test2 = (float) d2.getScalar(i,j).element();
                assertEquals(test1,test2,1e-6);
            }
        }
    }


    protected void verifyElements(DoubleMatrix d,INDArray d2) {
        if(d.isVector() && d2.isVector())
            for(int j = 0; j < d2.length(); j++) {
                float test1 = (float) d.get(j);
                float test2 = (float) d2.getScalar(j).element();
                assertEquals(test1,test2,1e-6);
            }

        else {
            for(int i = 0; i < d.rows; i++) {
                for(int j = 0; j < d.columns; j++) {
                    float test1 = (float) d.get(i,j);
                    float test2 = (float) d2.getScalar(i,j).element();
                    assertEquals(test1,test2,1e-6);
                }
            }
        }

    }

}
