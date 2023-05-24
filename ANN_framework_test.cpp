#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<fstream>
#include "ANN_framework.h"

using namespace std ;


int main()
{
    matrix matin(20,270);
    matrix matout(4,270);
    matrix matin1(20,270);
    matrix matout1(4,270);
    int c,d;
    double sum=0;
    fstream file;
    fstream file1;
    //----------------------------------------------------------------------------------------------------------
    //read output data matrix from folder
    file.open("dnnin.txt", ios::in);
    double test;
    //read input data matrix from folder
    for(c=0;c<20;c++)
    {
        for(d=0;d<270;d++)
        {
            file>>test;
            matin.ptr[c][d] = test;
        }
    }
    //----------------------------------------------------------------------------------------------------------
    //read output data and convert to classes based on percentages
    file1.open("dnnout.txt", ios::in);

    for(d=0;d<270;d++)
        {
            file1>>test;
            if(test>=0 && test <=10)
            {
                matout.ptr[0][d] = 1;
                matout.ptr[1][d] = 0;
                matout.ptr[2][d] = 0;
                matout.ptr[3][d] = 0;
            }
            else if(test>10 && test<=20)
            {
                matout.ptr[0][d] = 0;
                matout.ptr[1][d] = 1;
                matout.ptr[2][d] = 0;
                matout.ptr[3][d] = 0;
            }
            else if(test>20 && test<=30)
            {
                matout.ptr[0][d] = 0;
                matout.ptr[1][d] = 0;
                matout.ptr[2][d] = 1;
                matout.ptr[3][d] = 0;
            }
            else if(test>30 && test<=40)
            {
                matout.ptr[0][d] = 0;
                matout.ptr[1][d] = 0;
                matout.ptr[2][d] = 0;
                matout.ptr[3][d] = 1;
            }
        }
    //----------------------------------------------------------------------------------------------------------
    //randomise the data as minibatch gradient descent requires unique data shuffling
    //1. Create the random indices
    int flag, random[270];

    for(c=0;c<270;c++)
    {
        random[c] = 500;
    }

    for(c=0;c<270;c++)
    {
        flag=0;
        while(flag==0)
        {
            d = rand();
            d = d%270;
            flag = check(random,d);
        }
        random[c] = d;
    }
    cout<<"\nIndices Randomised!\n------------------------------------------------------------------------------"<<endl;
    //2. Shuffle the Input and Output Data According to the Randomised Indices
    for(c=0;c<270;c++)
    {
        for(d=0;d<20;d++)
        {
            matin1.ptr[d][c] = matin.ptr[d][random[c]];
        }
        for(d=0;d<4;d++)
        {
            matout1.ptr[d][c] = matout.ptr[d][random[c]];
        }
    }
    //----------------------------------------------------------------------------------------------------------
    //PROVIDE INPUTS TO THE NEURAL NETWORK AND TRAIN IT
    matrix intonetx(20,260);//training + cross validation input matrix
    matrix testnetx(20,10);//training + cross validation label matrix
    matrix intonety(4,260);//test input matrix
    matrix testnety(4,10);//test label matrix
    copymat(&matin1, &intonetx, 0, 259);
    copymat(&matout1, &intonety, 0, 259);
    copymat(&matin1, &testnetx, 260, 269);
    copymat(&matout1, &testnety, 260, 269);
    //input matrix format = (nx,m)-> nx: Features, m: Number of samples/examples
    //output matrix dimensions = (classes,m) = classes = classification column vectors for m examples
    //number of layers, Number of Hidden Units, Minibatch size, Alpha(Learning Rate, Lamda(L2 Regularisation Parameter): ANY;
    //Activations: 0-ReLu, 1-Sigmoid, 2-Tanh, 3-Softmax(Implemented as default for Last/Classification Layer)
    net net1(&intonetx,&intonety, /*trainsize:*/210,/*nlayers:*/ 5,/*hidden:*/ 10,/*activation:*/ 1,/*minibatch_size:*/ 10, /*epochs*/10000,/*alpha*/ -0.01,/*lamda*/ -6.33, /*dropout rate*/0.8);
    net1.trainnet();
    net1.test(&testnetx, &testnety);
    return 0;
}
