package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.jcublas.complex.ComplexDouble;
import org.nd4j.linalg.jcublas.complex.ComplexFloat;
import org.nd4j.linalg.jcublas.complex.JCublasComplexNDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;

/**
 * Created by mjk on 8/21/14.
 */
public class JCublasNDArrayFactory extends BaseNDArrayFactory {

    public JCublasNDArrayFactory(String dtype,Character order) {
        super(dtype,order);
    }

    /**
     * Create float
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    @Override
    public IComplexFloat createFloat(float real, float imag) {
        return new ComplexFloat(real,imag);
    }

    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    @Override
    public IComplexDouble createDouble(double real, double imag) {
        return new ComplexDouble(real,imag);
    }

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    @Override
    public INDArray create(double[][] data) {
        return new JCublasNDArray(data);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(INDArray arr) {
        return new JCublasComplexNDArray(arr);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data  the data to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        if(order == FORTRAN)
            return new JCublasComplexNDArray(data,shape,ArrayUtil.calcStridesFortran(shape));
        else
            return new JCublasComplexNDArray(data,shape);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs  the arr to wrap
     * @param shape
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    @Override
    public IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        if(order == FORTRAN)
            return new JCublasComplexNDArray(arrs,shape,ArrayUtil.calcStridesFortran(shape));

        return new JCublasComplexNDArray(arrs,shape);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
        return new JCublasComplexNDArray(data,shape,stride,offset);
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, char ordering) {
        return null;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset) {
        return new JCublasNDArray(data,shape,stride,offset);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
        return new JCublasComplexNDArray(ArrayUtil.floatCopyOf(data),shape,stride,offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        return new JCublasNDArray(data,shape,stride,offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param list
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        if(order == FORTRAN)
            return new JCublasNDArray(list,shape, ArrayUtil.calcStridesFortran(shape));
        else
            return new JCublasNDArray(list,shape);
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new JCublasComplexNDArray(ArrayUtil.floatCopyOf(data),shape,stride,offset,ordering);
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering) {
        return new JCublasComplexNDArray(ArrayUtil.floatCopyOf(data),shape,offset,ordering);
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return new JCublasComplexNDArray(ArrayUtil.floatCopyOf(data),shape,offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset) {
        return new JCublasNDArray(data,shape,offset);
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering) {
        return new JCublasComplexNDArray(data,shape, NDArrays.getComplexStrides(shape, ordering),offset,ordering);
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        return new JCublasComplexNDArray(data,shape,offset);
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new JCublasComplexNDArray(data,shape,stride,offset,ordering);
    }

    @Override
    public INDArray create(float[][] floats) {
        return new JCublasNDArray(floats);
    }

    @Override
    public IComplexNDArray createComplex(float[] dim) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return new JCublasNDArray(data,shape,stride,offset,ordering);
    }
}
