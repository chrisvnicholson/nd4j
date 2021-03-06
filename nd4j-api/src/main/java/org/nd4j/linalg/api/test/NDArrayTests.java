package org.nd4j.linalg.api.test;


import org.nd4j.linalg.api.ndarray.DimensionSlice;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.reduceops.Ops;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.Shape;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public abstract class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);
    private INDArray n = NDArrays.create(NDArrays.linspace(1,8,8).data(),new int[]{2,2,2});



    @Before
    public void before() {
        NDArrays.factory().setOrder('c');
    }

    @After
    public void after() {
        NDArrays.factory().setOrder('c');
    }

    @Test
    public void testScalarOps() {
        INDArray n = NDArrays.create(NDArrays.ones(27).data(),new int[]{3,3,3});
        assertEquals(27d,n.length(),1e-1);
        n.checkDimensions(n.addi(NDArrays.scalar(1d)));
        n.checkDimensions(n.subi(NDArrays.scalar(1.0d)));
        n.checkDimensions(n.muli(NDArrays.scalar(1.0d)));
        n.checkDimensions(n.divi(NDArrays.scalar(1.0d)));

        n = NDArrays.create(NDArrays.ones(27).data(),new int[]{3,3,3});
        assertEquals(27,(float) n.sum(Integer.MAX_VALUE).element(),1e-1);
        INDArray a = n.slice(2);
        assertEquals(true,Arrays.equals(new int[]{3,3},a.shape()));

    }


    @Test
    public void testLinearViewGetAndPut() {
        INDArray test = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray linear = test.linearView();
        linear.putScalar(2,6);
        linear.putScalar(3,7);
        assertEquals(6,linear.get(2),1e-1);
        assertEquals(7,linear.get(3),1e-1);
    }


    @Test
    public void testGetIndices() {
        /*[[[1.0 ,13.0],[5.0 ,17.0],[9.0 ,21.0]],[[2.0 ,14.0],[6.0 ,18.0],[10.0 ,22.0]],[[3.0 ,15.0],[7.0 ,19.0],[11.0 ,23.0]],[[4.0 ,16.0],[8.0 ,20.0],[12.0 ,24.0]]]*/
        NDArrays.factory().setOrder('f');
        INDArray test = NDArrays.linspace(1,24,24).reshape(new int[]{4,3,2});
        NDArrayIndex oneTwo = NDArrayIndex.interval(1, 2);
        NDArrayIndex twoToThree = NDArrayIndex.interval(1,3);
        INDArray get = test.get(oneTwo,twoToThree);
        assertTrue(Arrays.equals(new int[]{1,2,2},get.shape()));
        assertEquals(NDArrays.create(new float[]{6,10,18,22},new int[]{1,2,2}),get);

        INDArray anotherGet = NDArrays.create(new float[]{6,7,10,11,18,19,22,23},new int[]{2,1,2});
        INDArray test2 = test.get(NDArrayIndex.interval(1,3),NDArrayIndex.interval(1,4));
        assertEquals(5,test2.offset());
        //offset is off: should be 5
        assertTrue(Arrays.equals(new int[]{2,1,2},test2.shape()));
        assertEquals(test2,anotherGet);


    }


    @Test
    public void testGetIndicesVector() {
        INDArray line = NDArrays.linspace(1,4,4);
        INDArray test = NDArrays.create(new float[]{2,3});
        INDArray result = line.get(NDArrayIndex.interval(1, 3));
        assertEquals(test,result);
    }

    @Test
    public void testGetIndices2d() {
        NDArrays.factory().setOrder('f');

        INDArray twoByTwo = NDArrays.linspace(1,6,6).reshape(3,2);
        INDArray firstRow = twoByTwo.getRow(0);
        INDArray secondRow = twoByTwo.getRow(1);
        INDArray firstAndSecondRow = twoByTwo.getRows(new int[]{1,2});
        INDArray firstRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(0,1));
        assertEquals(firstRow,firstRowViaIndexing);
        INDArray secondRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(1,2));
        assertEquals(secondRow,secondRowViaIndexing);
        INDArray individualElement = twoByTwo.get(NDArrayIndex.interval(1,2),NDArrayIndex.interval(1,2));
        assertEquals(NDArrays.create(new float[]{5}),individualElement);

        INDArray firstAndSecondRowTest = twoByTwo.get(NDArrayIndex.interval(1, 3));
        assertEquals(firstAndSecondRow, firstAndSecondRowTest);


    }

    @Test
    public void testGetVsGetScalar() {
        INDArray a = NDArrays.linspace(1,4,4).reshape(2,2);
        float element = a.get(0,1);
        float element2 = (float) a.getScalar(0,1).element();
        assertEquals(element,element2,1e-1);
        NDArrays.factory().setOrder('f');
        INDArray a2 = NDArrays.linspace(1,4,4).reshape(2,2);
        float element23 = a2.get(0,1);
        float element22 = (float) a2.getScalar(0,1).element();
        assertEquals(element23,element22,1e-1);

    }

    @Test
    public void testDivide() {
        INDArray two = NDArrays.create(new float[]{2,2,2,2});
        INDArray div = two.div(two);
        assertEquals(NDArrays.ones(4),div);

        INDArray half = NDArrays.create(new float[]{0.5f,0.5f,0.5f,0.5f},new int[]{2,2});
        INDArray divi = NDArrays.create(new float[]{0.3f,0.6f,0.9f,0.1f}, new int[]{2,2});
        INDArray assertion = NDArrays.create(new float[]{1.6666666f,0.8333333f,0.5555556f,5},new int[]{2,2});
        INDArray result = half.div(divi);
        assertEquals(assertion,result);
    }


    @Test
    public void testSigmoid() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4});
        INDArray assertion = NDArrays.create(new float[]{ 0.73105858f , 0.88079708f , 0.95257413f,  0.98201379f});
        INDArray sigmoid = Transforms.sigmoid(n);
        assertEquals(assertion,sigmoid);
    }

    @Test
    public void testNeg() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4});
        INDArray assertion = NDArrays.create(new float[]{-1,-2,-3,-4});
        INDArray neg = Transforms.neg(n);
        assertEquals(assertion,neg);

    }

    @Test
    public void testNorm2() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4});
        float assertion = 5.47722557505f;
        assertEquals(assertion,n.norm2(Integer.MAX_VALUE).get(0),1e-1);
    }

    @Test
    public void testExp() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4});
        INDArray assertion = NDArrays.create(new float[]{2.71828183f,   7.3890561f ,  20.08553692f,  54.59815003f});
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion,exped);
    }




    @Test
    public void testSlices() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        INDArray a = NDArrays.scalar(1.0);
        assertEquals(true,a.isScalar());

        INDArray n = NDArrays.create(new float[]{1.0f},new int[]{1,1});
        assertEquals(n,a);
        assertTrue(n.isScalar());
    }

    @Test
    public void testWrap() {
        int[] shape = {2,4};
        INDArray d = NDArrays.linspace(1,8,8).reshape(shape[0],shape[1]);
        INDArray n =d;
        assertEquals(d.rows(),n.rows());
        assertEquals(d.columns(),n.columns());

        INDArray vector = NDArrays.linspace(1,3,3);
        INDArray testVector = vector;
        for(int i = 0; i < vector.length(); i++)
            assertEquals((float) vector.getScalar(i).element(),(float) testVector.getScalar(i).element(),1e-1);
        assertEquals(3,testVector.length());
        assertEquals(true,testVector.isVector());
        assertEquals(true,Shape.shapeEquals(new int[]{3},testVector.shape()));

        INDArray row12 = NDArrays.linspace(1,2,2).reshape(2,1);
        INDArray row22 = NDArrays.linspace(3,4,2).reshape(1,2);

        assertEquals(row12.rows(),2);
        assertEquals(row12.columns(),1);
        assertEquals(row22.rows(),1);
        assertEquals(row22.columns(),2);



    }

    @Test
    public void testGetRowFortran() {
        NDArrays.factory().setOrder('f');
        INDArray n = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        INDArray column = NDArrays.create(new float[]{1,3});
        INDArray column2 = NDArrays.create(new float[]{2,4});
        INDArray testColumn = n.getRow(0);
        INDArray testColumn1 = n.getRow(1);
        assertEquals(column,testColumn);
        assertEquals(column2,testColumn1);
        NDArrays.factory().setOrder('c');

    }

    @Test
    public void testGetColumnFortran() {
        NDArrays.factory().setOrder('f');
        INDArray n = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        INDArray column = NDArrays.create(new float[]{1,2});
        INDArray column2 = NDArrays.create(new float[]{3,4});
        INDArray testColumn = n.getColumn(0);
        INDArray testColumn1 = n.getColumn(1);
        assertEquals(column,testColumn);
        assertEquals(column2,testColumn1);
        NDArrays.factory().setOrder('c');

    }


    @Test
    public void testVectorInit() {
        float[] data = NDArrays.linspace(1,4,4).data();
        INDArray arr = NDArrays.create(data,new int[]{4});
        assertEquals(true,arr.isRowVector());
        INDArray arr2 = NDArrays.create(data,new int[]{1,4});
        assertEquals(true,arr2.isRowVector());

        INDArray columnVector = NDArrays.create(data,new int[]{4,1});
        assertEquals(true,columnVector.isColumnVector());
    }













    @Test
    public void testColumns() {
        INDArray arr = NDArrays.create(new int[]{3,2});
        INDArray column2 = arr.getColumn(0);
        assertEquals(true,Shape.shapeEquals(new int[]{3,1}, column2.shape()));
        INDArray column = NDArrays.create(new float[]{1,2,3},new int[]{3});
        arr.putColumn(0,column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column,firstColumn);


        INDArray column1 = NDArrays.create(new float[]{4,5,6},new int[]{3});
        arr.putColumn(1,column1);
        assertEquals(true, Shape.shapeEquals(new int[]{3,1}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1,testRow1);



        INDArray evenArr = NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
        INDArray put = NDArrays.create(new float[]{5,6},new int[]{2});
        evenArr.putColumn(1,put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put,testColumn);



        INDArray n = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = NDArrays.create(new float[]{1,3},new int[]{2});
        assertEquals(column23,column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = NDArrays.create(new float[]{2,4},new int[]{2});
        assertEquals(column0,column01);



    }


    @Test
    public void testPutRow() {
        INDArray d = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = reshape(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 2;
        float dFirst = d.get(0, 1);
        assertEquals(nFirst,dFirst,1e-1);
        assertEquals(true,Arrays.equals(d.data(),n.data()));
        assertEquals(true,Arrays.equals(new int[]{2,2},n.shape()));

        INDArray newRow = NDArrays.linspace(5,6,2);
        n.putRow(0,newRow);
        d.putRow(0,newRow);



        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(),testRow.length());
        assertEquals(true, Shape.shapeEquals(new int[]{2}, testRow.shape()));



        INDArray nLast = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = NDArrays.create(new float[]{3,4},new int[]{2});
        assertEquals(row,row1);



        INDArray arr = NDArrays.create(new int[]{3,2});
        INDArray evenRow = NDArrays.create(new float[]{1,2},new int[]{2});
        arr.putRow(0,evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2},firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow,testRowEven);


        INDArray row12 = NDArrays.create(new float[]{5,6},new int[]{2});
        arr.putRow(1,row12);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12,testRow1);


        INDArray multiSliceTest = NDArrays.create(NDArrays.linspace(1,16,16).data(),new int[]{4,2,2});
        INDArray test = NDArrays.create(new float[]{7,8},new int[]{2});
        INDArray test2 = NDArrays.create(new float[]{9,10},new int[]{2});

        assertEquals(test,multiSliceTest.slice(1).getRow(1));
        assertEquals(test2,multiSliceTest.slice(1).getRow(2));

    }

    @Test
    public void testOrdering() {
        //c ordering first
        NDArrays.factory().setOrder('c');
        NDArrays.factory().setDType("float");

        INDArray  data = NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
        assertEquals(2.0,(float) data.getScalar(0,1).element(),1e-1);
        NDArrays.factory().setOrder('f');

        INDArray data2 = NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
        assertNotEquals(data2.getScalar(0,1),data.getScalar(0,1));
        NDArrays.factory().setOrder('c');

    }





    @Test
    public void testSum() {
        INDArray n = NDArrays.create(NDArrays.linspace(1,8,8).data(),new int[]{2,2,2});
        INDArray test = NDArrays.create(new float[]{3,7,11,15},new int[]{2,2});
        INDArray sum = n.sum(n.shape().length - 1);
        assertEquals(test,sum);
    }



    @Test
    public void testMmulF() {
        NDArrays.factory().setOrder('f');

        float[] data = NDArrays.linspace(1,10,10).data();
        INDArray n = NDArrays.create(data,new int[]{10});
        INDArray transposed = n.transpose();
        assertEquals(true,n.isRowVector());
        assertEquals(true,transposed.isColumnVector());

        INDArray d = NDArrays.create(Arrays.copyOf(n.data(),n.data().length),new int[]{n.rows(),n.columns()});


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = NDArrays.scalar(385);
        assertEquals(scalar,innerProduct);

    }


    @Test
    public void testMmul() {

        NDArrays.factory().setOrder('c');

        float[] data = NDArrays.linspace(1,10,10).data();
        INDArray n = NDArrays.create(data,new int[]{10});
        INDArray transposed = n.transpose();
        assertEquals(true,n.isRowVector());
        assertEquals(true,transposed.isColumnVector());

        INDArray d = NDArrays.create(n.rows(),n.columns());
        d.setData(n.data());


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = NDArrays.scalar(385);
        assertEquals(scalar,innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10,10},outerProduct.shape()));


        INDArray testMatrix = NDArrays.create(data,new int[]{5,2});
        INDArray row1 = testMatrix.getRow(0).transpose();
        INDArray row2 = testMatrix.getRow(1);
        INDArray row12 = NDArrays.linspace(1,2,2).reshape(2,1);
        INDArray row22 = NDArrays.linspace(3,4,2).reshape(1,2);

        INDArray row122 = row12;
        INDArray row222 = row22;
        INDArray rowResult2 = row122.mmul(row222);



        INDArray d3 = NDArrays.create(new float[]{1,2}).reshape(2,1);
        INDArray d4 = NDArrays.create(new float[]{3,4});
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = NDArrays.create(new float[][]{{3,4},{6,8}});

        assertEquals(result,resultNDArray);




        INDArray three = NDArrays.create(new float[]{3,4},new int[]{2});
        INDArray test = NDArrays.create(NDArrays.linspace(1,30,30).data(),new int[]{3,5,2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three,sliceRow);

        INDArray twoSix = NDArrays.create(new float[]{2,6},new int[]{2,1});
        INDArray threeTwoSix = three.mmul(twoSix);

        INDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix,sliceRowTwoSix);


        INDArray vectorVector = NDArrays.create(new float[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225
        },new int[]{16,16});


        INDArray n1 = NDArrays.create(NDArrays.linspace(0,15,16).data(),new int[]{16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(vectorVector,testVectorVector);



    }

    @Test
    public void testRowsColumns() {
        float[] data = NDArrays.linspace(1,6,6).data();
        INDArray rows = NDArrays.create(data,new int[]{2,3});
        assertEquals(2,rows.rows());
        assertEquals(3,rows.columns());

        INDArray columnVector = NDArrays.create(data,new int[]{6,1});
        assertEquals(6,columnVector.rows());
        assertEquals(1,columnVector.columns());
        INDArray rowVector = NDArrays.create(data,new int[]{6});
        assertEquals(1,rowVector.rows());
        assertEquals(6,rowVector.columns());
    }


    @Test
    public void testTranspose() {
        INDArray n = NDArrays.create(NDArrays.ones(100).data(),new int[]{5,5,4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(),transpose.length());
        assertEquals(true,Arrays.equals(new int[]{4,5,5},transpose.shape()));

        INDArray rowVector = NDArrays.linspace(1,10,10);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray transposed = NDArrays.create(new float[]{1,3,2,4},new int[]{2,2});
        assertEquals(transposed,linspaced.transpose());

        NDArrays.factory().setOrder('f');
        linspaced = NDArrays.linspace(1,4,4).reshape(2,2);
        //fortran ordered
        INDArray transposed2 = NDArrays.create(new float[]{1,3,2,4},new int[]{2,2});
        transposed = linspaced.transpose();
        assertEquals(transposed,transposed2);
        NDArrays.factory().setOrder('c');





    }

    @Test
    public void testPutSlice() {
        INDArray n = NDArrays.create(NDArrays.ones(27).data(),new int[]{3,3,3});
        INDArray newSlice = NDArrays.zeros(3,3);
        n.putSlice(0,newSlice);
        assertEquals(newSlice,n.slice(0));


    }

    @Test
    public void testPermute() {
        INDArray n = NDArrays.create(NDArrays.linspace(1,20,20).data(),new int[]{5,4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(new int[]{1,0});
        assertEquals(permute,transpose);
        assertEquals(transpose.length(),permute.length(),1e-1);


        INDArray toPermute = NDArrays.create(NDArrays.linspace(0,7,8).data(),new int[]{2,2,2});
        INDArray permuted = toPermute.permute(new int[]{2,1,0});
        INDArray assertion = NDArrays.create(new float[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(permuted,assertion);

    }

    @Test
    public void testSlice() {
        assertEquals(8,n.length());
        assertEquals(true,Arrays.equals(new int[]{2,2,2},n.shape()));
        INDArray slice = n.slice(0);
        assertEquals(true, Arrays.equals(new int[]{2, 2}, slice.shape()));

        INDArray slice1 = n.slice(1);
        assertNotEquals(slice,slice1);

        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        INDArray slice0 = NDArrays.create(new float[]{1,2,3,4,5,6},new int[]{3,2});
        INDArray slice2 = NDArrays.create(new float[]{7,8,9,10,11,12},new int[]{3,2});

        INDArray testSlice0 = arr.slice(0);
        INDArray testSlice1 = arr.slice(1);

        assertEquals(slice0,testSlice0);
        assertEquals(slice2,testSlice1);







    }

    @Test
    public void testSwapAxes() {
        INDArray n = NDArrays.create(NDArrays.linspace(0,7,8).data(),new int[]{2,2,2});
        INDArray assertion = n.permute(new int[]{2,1,0});
        float[] data = assertion.data();
        INDArray validate = NDArrays.create(new float[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(validate,assertion);



    }




    @Test
    public void testLinearIndex() {
        INDArray n = NDArrays.create(NDArrays.linspace(1,8,8).data(),new int[]{8});
        for(int i = 0; i < n.length(); i++) {
            int linearIndex = n.linearIndex(i);
            assertEquals(i ,linearIndex);
            float d =  (float) n.getScalar(i).element();
            assertEquals(i + 1,d,1e-1);
        }
    }

    @Test
    public void testSliceConstructor() {
        List<INDArray> testList = new ArrayList<>();
        for(int i = 0; i < 5; i++)
            testList.add(NDArrays.scalar(i + 1));

        INDArray test = NDArrays.create(testList,new int[]{testList.size()});
        INDArray expected = NDArrays.create(new float[]{1,2,3,4,5},new int[]{5});
        assertEquals(expected,test);
    }





    @Test
    public void testVectorDimension() {
        INDArray test = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        final AtomicInteger count = new AtomicInteger(0);
        //row wise
        test.iterateOverDimension(1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                INDArray test = (INDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray firstDimension = NDArrays.create(new float[]{1,2},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = NDArrays.create(new float[]{3,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                INDArray test = nd;
                if(count.get() == 0) {
                    INDArray firstDimension = NDArrays.create(new float[]{1,2},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = NDArrays.create(new float[]{3,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

        },false);



        count.set(0);

        //columnwise
        test.iterateOverDimension(0,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                log.info("Operator " + nd);
                INDArray test = (INDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray firstDimension = NDArrays.create(new float[]{1,3},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = NDArrays.create(new float[]{2,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                log.info("Operator " + nd);
                INDArray test = nd;
                if(count.get() == 0) {
                    INDArray firstDimension = NDArrays.create(new float[]{1,3},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = NDArrays.create(new float[]{2,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

        },false);




    }



    @Test
    public void testDimension() {
        INDArray test = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        //row
        INDArray slice0 = test.slice(0,1);
        INDArray slice02 = test.slice(1,1);

        INDArray assertSlice0 = NDArrays.create(new float[]{1,2});
        INDArray assertSlice02 = NDArrays.create(new float[]{3,4});
        assertEquals(assertSlice0,slice0);
        assertEquals(assertSlice02,slice02);

        //column
        INDArray assertSlice1 = NDArrays.create(new float[]{1,3});
        INDArray assertSlice12 = NDArrays.create(new float[]{2,4});


        INDArray slice1 = test.slice(0,0);
        INDArray slice12 = test.slice(1,0);


        assertEquals(assertSlice1,slice1);
        assertEquals(assertSlice12,slice12);



        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        INDArray firstSliceFirstDimension = arr.slice(0,1);
        INDArray secondSliceFirstDimension = arr.slice(1,1);

        INDArray firstSliceFirstDimensionAssert = NDArrays.create(new float[]{1,2,7,8,13,14,19,20});
        INDArray secondSliceFirstDimension2Test = firstSliceFirstDimensionAssert.add(1);
        assertEquals(secondSliceFirstDimension,secondSliceFirstDimension);


    }


    @Test
    public void testReshape() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        INDArray reshaped = arr.reshape(new int[]{2,3,4});
        assertEquals(arr.length(),reshaped.length());
        assertEquals(true,Arrays.equals(new int[]{4,3,2},arr.shape()));
        assertEquals(true,Arrays.equals(new int[]{2,3,4},reshaped.shape()));

        INDArray n2 = NDArrays.create(NDArrays.linspace(1, 30, 30).data(),new int[]{3,5,2});
        INDArray swapped   = n2.swapAxes(n2.shape().length - 1,1);
        INDArray firstSlice2 = swapped.slice(0).slice(0);
        INDArray oneThreeFiveSevenNine = NDArrays.create(new float[]{1,3,5,7,9});
        assertEquals(firstSlice2,oneThreeFiveSevenNine);
        INDArray raveled = oneThreeFiveSevenNine.reshape(5,1);
        INDArray raveledOneThreeFiveSevenNine = oneThreeFiveSevenNine.reshape(5,1);
        assertEquals(raveled,raveledOneThreeFiveSevenNine);



        INDArray firstSlice3 = swapped.slice(0).slice(1);
        INDArray twoFourSixEightTen = NDArrays.create(new float[]{2,4,6,8,10});
        assertEquals(firstSlice2,oneThreeFiveSevenNine);
        INDArray raveled2 = twoFourSixEightTen.reshape(5,1);
        INDArray raveled3 = firstSlice3.reshape(5,1);
        assertEquals(raveled2,raveled3);


    }


    @Test
    public void reduceTest() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        INDArray reduced = arr.reduce(Ops.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(Ops.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(Ops.DimensionOp.MAX,2);
        log.info("Reduced " + reduced);


    }




    @Test
    public void testColumnVectorOpsFortran() {
        NDArrays.factory().setOrder('f');
        INDArray twoByTwo = NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
        INDArray toAdd = NDArrays.create(new float[]{1,2},new int[]{2,1});
        twoByTwo.addiColumnVector(toAdd);
        INDArray assertion = NDArrays.create(new float[]{2,4,4,6},new int[]{2,2});
        assertEquals(assertion,twoByTwo);



    }


    @Test
    public void testMeans() {
        INDArray a = NDArrays.linspace(1,4,4).reshape(2,2);
        assertEquals(NDArrays.create(new float[]{2,3}),a.mean(0));
        assertEquals(NDArrays.create(new float[]{1.5f,3.5f}),a.mean(1));
        assertEquals(2.5,(float) a.mean(Integer.MAX_VALUE).element(),1e-1);

    }


    @Test
    public void testSums() {
        INDArray a = NDArrays.linspace(1,4,4).reshape(2,2);
        assertEquals(NDArrays.create(new float[]{4,6}),a.sum(0));
        assertEquals(NDArrays.create(new float[]{3,7}),a.sum(1));
        assertEquals(10,(float) a.sum(Integer.MAX_VALUE).element(),1e-1);


    }


    @Test
    public void testCumSum() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4}, new int[]{4});
        INDArray cumSumAnswer = NDArrays.create(new float[]{1,3,6,10}, new int[]{4});
        INDArray cumSumTest = n.cumsum(0);
        assertEquals(cumSumAnswer,cumSumTest);

        INDArray n2 = NDArrays.linspace(1,24,24).reshape(new int[]{4,3,2});
        INDArray cumSumCorrect2 = NDArrays.create(new double[]{1.0,3.0,6.0,10.0,15.0,21.0,28.0,36.0,45.0,55.0,66.0,78.0,91.0,105.0,120.0,136.0,153.0,171.0,190.0,210.0,231.0,253.0,276.0,300.0},new int[]{24});
        INDArray cumSumTest2 = n2.cumsum(n2.shape().length - 1);
        assertEquals(cumSumCorrect2,cumSumTest2);

        INDArray axis0assertion = NDArrays.create(new float[]{1,2,3,4,5,6,8,10,12,14,16,18,21,24,27,30,33,36,40,44,48,52,56,60},n2.shape());
        INDArray axis0Test = n2.cumsum(0);
        assertEquals(axis0assertion,axis0Test);

    }


    @Test
    public void testRSubi() {
        INDArray n2 = NDArrays.ones(2);
        INDArray n2Assertion = NDArrays.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion,nRsubi);
    }

    @Test
    public void testRDivi() {
        INDArray n2 = NDArrays.valueArrayOf(new int[]{2},4);
        INDArray n2Assertion = NDArrays.valueArrayOf(new int[]{2},0.5);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion,nRsubi);
    }


    @Test
    public void testVectorAlongDimension() {
        INDArray arr = NDArrays.linspace(1,24,24).reshape(new int[]{4,3,2});
        INDArray assertion = NDArrays.create(new float[]{1,2}, new int[]{2});
        assertEquals(NDArrays.create(new float[]{3,4},new int[]{2}),arr.vectorAlongDimension(1,2));
        assertEquals(assertion,arr.vectorAlongDimension(0,2));
        assertEquals(arr.vectorAlongDimension(0,1),NDArrays.create(new float[]{1,3,5}));

        INDArray testColumn2Assertion = NDArrays.create(new float[]{7,9,11});
        INDArray testColumn2 = arr.vectorAlongDimension(1,1);

        assertEquals(testColumn2Assertion,testColumn2);


        INDArray testColumn3Assertion = NDArrays.create(new float[]{13,15,17});
        INDArray testColumn3 = arr.vectorAlongDimension(2,1);
        assertEquals(testColumn3Assertion,testColumn3);


        INDArray v1= NDArrays.linspace(1,4,4).reshape(new int[]{2,2});
        INDArray testColumnV1 = v1.vectorAlongDimension(0,0);
        INDArray testColumnV1Assertion = NDArrays.create(new float[]{1,3});
        assertEquals(testColumnV1Assertion,testColumnV1);

        INDArray testRowV1 = v1.vectorAlongDimension(1,0);
        INDArray testRowV1Assertion = NDArrays.create(new float[]{2,4});
        assertEquals(testRowV1Assertion,testRowV1);


        INDArray lastAxis = arr.vectorAlongDimension(0,2);
        assertEquals(assertion,lastAxis);






    }

    @Test
    public void testSquareMatrix() {
        INDArray n = NDArrays.create(NDArrays.linspace(1,8,8).data(),new int[]{2,2,2});
        INDArray eightFirstTest = n.vectorAlongDimension(0,2);
        INDArray eightFirstAssertion = NDArrays.create(new float[]{1,2},new int[]{2});
        assertEquals(eightFirstAssertion,eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1,2);
        INDArray eightFirstTestSecondAssertion = NDArrays.create(new float[]{3,4});
        assertEquals(eightFirstTestSecondAssertion,eightFirstTestSecond);

    }

    @Test
    public void testNumVectorsAlongDimension() {
        INDArray arr = NDArrays.linspace(1,24,24).reshape(new int[]{4,3,2});
        assertEquals(12,arr.vectorsAlongDimension(2));
    }


    @Test
    public void testGetScalar() {
        INDArray n = NDArrays.create(new float[]{1,2,3,4},new int[]{4});
        assertTrue(n.isVector());
        for(int i = 0; i < n.length(); i++) {
            INDArray scalar = NDArrays.scalar((float) i + 1);
            assertEquals(scalar,n.getScalar(i));
        }

        NDArrays.factory().setOrder('f');
        n = NDArrays.create(new float[]{1,2,3,4},new int[]{4});
        for(int i = 0; i < n.length(); i++) {
            INDArray scalar = NDArrays.scalar((float) i + 1);
            assertEquals(scalar,n.getScalar(i));
        }


        INDArray twoByTwo = NDArrays.create(new float[][]{{1,2},{3,4}});
        INDArray column = twoByTwo.getColumn(0);
        assertEquals(NDArrays.create(new float[]{1,3}),column);
        assertEquals(1,column.get(0),1e-1);
        assertEquals(3,column.get(1),1e-1);
        assertEquals(NDArrays.scalar(1),column.getScalar(0));
        assertEquals(NDArrays.scalar(3),column.getScalar(1));


    }

    @Test
    public void testGetMulti() {
        assertEquals(8,n.length());
        assertEquals(true,Arrays.equals(ArrayUtil.of(2, 2, 2),n.shape()));
        float val = (float) n.getScalar(1,1,1).element();
        assertEquals(8.0,val,1e-6);
    }


    @Test
    public void testGetRowOrdering() {
        INDArray row1 = NDArrays.linspace(1,4,4).reshape(2,2);
        NDArrays.factory().setOrder('f');
        INDArray row1Fortran = NDArrays.linspace(1,4,4).reshape(2,2);
        assertNotEquals(row1.get(0,1),row1Fortran.get(0,1),1e-1);
        NDArrays.factory().setOrder('c');
    }


    @Test
    public void testPutRowGetRowOrdering() {
        INDArray row1 = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray put = NDArrays.create(new float[]{5,6});
        row1.putRow(1,put);

        NDArrays.factory().setOrder('f');

        INDArray row1Fortran = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray putFortran = NDArrays.create(new float[]{5,6});
        row1Fortran.putRow(1,putFortran);
        assertNotEquals(row1,row1Fortran);
        INDArray row1CTest = row1.getRow(1);
        INDArray row1FortranTest = row1Fortran.getRow(1);
        assertEquals(row1CTest,row1FortranTest);

        NDArrays.factory().setOrder('c');


    }



    @Test
    public void testPutRowFortran() {
        INDArray row1 = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray put = NDArrays.create(new float[]{5,6});
        row1.putRow(1,put);

        NDArrays.factory().setOrder('f');

        INDArray row1Fortran =NDArrays.create(new float[][]{{1,2},{3,4}});
        INDArray putFortran = NDArrays.create(new float[]{5,6});
        row1Fortran.putRow(1,putFortran);
        assertEquals(row1,row1Fortran);

        NDArrays.factory().setOrder('c');


    }


    @Test
    public void testElementWiseOps() {
        INDArray n1 = NDArrays.scalar(1);
        INDArray n2 = NDArrays.scalar(2);
        assertEquals(NDArrays.scalar(3),n1.add(n2));
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = NDArrays.scalar(3);
        INDArray n4 = NDArrays.scalar(4);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(NDArrays.scalar(1),subbed);
        assertEquals(NDArrays.scalar(12),mulled);
        assertEquals(NDArrays.scalar(1.333333333333333333333),div);
    }







    @Test
    public void testSlicing() {
        INDArray arr = n.slice(1, 1);
        // assertEquals(1,arr.shape().length());
        INDArray n2 = NDArrays.create(NDArrays.linspace(1,16,16).data(),new int[]{2,2,2,2});
        log.info("N2 shape " + n2.slice(1,1).slice(1));

    }


    @Test
    public void testEndsForSlices() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true,Arrays.equals(new int[]{5,11,17,23},endsForSlices));
    }


    @Test
    public void testFlatten() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,4,4).data(),new int[]{2,2});
        INDArray flattened = arr.ravel();
        assertEquals(arr.length(),flattened.length());
        assertEquals(true,Shape.shapeEquals(new int[]{1, arr.length()}, flattened.shape()));
        for(int i = 0; i < arr.length(); i++) {
            assertEquals(i + 1, flattened.get(i),1e-1);
        }
        assertTrue(flattened.isVector());


        INDArray n = NDArrays.create(NDArrays.ones(27).data(),new int[]{3,3,3});
        INDArray nFlattened = n.ravel();
        assertTrue(nFlattened.isVector());

        INDArray n1 = NDArrays.linspace(1,24,24);
        assertEquals(n1,NDArrays.linspace(1,24,24).reshape(new int[]{4,3,2}).ravel());



    }

    @Test
    public void testVectorDimensionMulti() {
        INDArray arr = NDArrays.create(NDArrays.linspace(1,24,24).data(),new int[]{4,3,2});
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(arr.shape().length - 1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                INDArray test =(INDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray answer = NDArrays.create(new float[]{1,7,13,19},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 1) {
                    INDArray answer = NDArrays.create(new float[]{2,8,14,20},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 2) {
                    INDArray answer = NDArrays.create(new float[]{3,9,15,21},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 3) {
                    INDArray answer = NDArrays.create(new float[]{4,10,16,22},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 4) {
                    INDArray answer = NDArrays.create(new float[]{5,11,17,23},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 5) {
                    INDArray answer = NDArrays.create(new float[]{6,12,18,24},new int[]{4});
                    assertEquals(answer,test);
                }


                count.incrementAndGet();
            }

            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                INDArray test =  nd;
                if(count.get() == 0) {
                    INDArray answer = NDArrays.create(new float[]{1,2},new int[]{2});
                    assertEquals(answer,test);
                }
                else if(count.get() == 1) {
                    INDArray answer = NDArrays.create(new float[]{3,4},new int[]{2});
                    assertEquals(answer,test);
                }
                else if(count.get() == 2) {
                    INDArray answer = NDArrays.create(new float[]{5,6},new int[]{2});
                    assertEquals(answer,test);
                }
                else if(count.get() == 3) {
                    INDArray answer = NDArrays.create(new float[]{7,8},new int[]{2});
                    assertEquals(answer,test);
                }
                else if(count.get() == 4) {
                    INDArray answer = NDArrays.create(new float[]{9,10},new int[]{2});
                    assertEquals(answer,test);
                }
                else if(count.get() == 5) {
                    INDArray answer = NDArrays.create(new float[]{11,12},new int[]{2});
                    assertEquals(answer,test);
                }


                count.incrementAndGet();
            }
        },false);
    }

}
