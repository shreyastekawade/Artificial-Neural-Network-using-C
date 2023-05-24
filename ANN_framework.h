#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<fstream>

using namespace std ;

/*Working of a Neural Network:

1. Take input matrix x : dim = (nx,m) - nx is number of features and m is number of examples

2. Initialize the weight matrices and constant matrices(b)based on user input
    Dimensions of the weight matrix of nth layer is = dim(wn) = (number of hidden units in nth layer) * (number of hidden units in (n-1)th layer)
    Dimensions of constant matrix of nth layer = dim(bn) = (number of hidden units in nth layer) * 1
    Dimensions of dwn = dimensions of w and dimensions of dbn = dimensions of bn

3. Steps for forward propagation:

    zn = ( wn * a(n-1) ) + bn
    an = g (zn)                     g(zn) can be = sigmoid (1/(1 + e^(-z))) , hyperbolic tan ((e^z - e^(-z))/(e^z = e^(-z))), relu (anything below zero
                                                                                                                                        is set to zero)
    last layer: y_hat = softmax(y_hat)

4. Cost Function for every element of last layer:
    cost = (1/(m* number of classes)) * sum(y * log10(y_hat) + (1-y) * log10(1-y_hat)) + (lambda/2m) * (elementwise square sum W)

5. Back Propagation
    For last layer, dzlast = y - y_pred
    For other layers, dzn = [w(n+1).T * dz(n+1)] * elementwise of derzn

    derivatives of zn =>
    sigmoid derivative = (1-g(z)) * g(z)
    tanh derivative = 1 - g(z)^2
    relu derivative = 1, for all z>=0
                    = 0, otherwise

    dwn = (1/m) * dzn * a(n-1).T
    dbn = (1/m) * horizontal sum(dzn)

    wn = wn - alpha * dwn - (lamba/m) * wn
    bn = bn - alpha * dbn - (lamba/m) * bn

    stages:
    1. Find dz of current layer
    2. Find dw and db of current layer
    3. use dw and db to find w and b of current layer

6. default values
    number of layers = 2
    number of hidden units = number of classes
    lambda = 0.01
    alpha (learning rate) = 0.001

7. Softmax function for last layer = F(x) = (e^x)/sum(e^x)

8. Precision of a Neural Network = TP/(TP + FP)

9. Recall of a Neural Network =TP/(TP + FN)

10. F1 Score as evaluation metric = (2 * Precision * Recall)/(Precision + Recall)

*/

#define e 2.71828

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class matrix
{
    public:
        int r;
        int c;
        double **ptr;

        matrix(void)
        {
            ptr = NULL;
        }
        matrix(int row,int col)
        {
            this->r = row, this->c = col;
            ptr = new double*[this->r];
            for (int k = 0; k < this->r; ++k)
                ptr[k] = new double[this->c];
            int c,d;
            double p;
            for(c=0;c<this->r;c++)
            {
                for(d=0;d<this->c;d++)
                {
                    p = (double)rand()/10000000;
                    ptr[c][d] = 4*((double)rand()/10000000);
                    if(p<0.0001)
                    {
                        ptr[c][d] = ptr[c][d] * (-1);
                    }
                }
            }
        }
        void operator = (int *dims)       //constructor for the layer
        {
            this->r = dims[0], this->c = dims[1];
            ptr = new double*[this->r];
            for (int k = 0; k < this->r; ++k)
                ptr[k] = new double[this->c];
            int c,d;
            double p;
            for(c=0;c<this->r;c++)
            {
                for(d=0;d<this->c;d++)
                {
                    p = (double)rand()/100000000;
                    ptr[c][d] = 4*((double)rand()/10000000);
                    if(p<0.0001)
                    {
                        ptr[c][d] = ptr[c][d] * (-1);
                    }
                }
            }
        }
        matrix(matrix *mat)
        {
            this->r = mat->r, this->c = mat->c;
            ptr = new double*[this->r];
            for (int k = 0; k < this->r; ++k)
                ptr[k] = new double[this->c];

            int c,d;
            double p;
            for(c=0;c<this->r;c++)
            {
                for(d=0;d<this->c;d++)
                {
                    p = (double)rand()/100000000;
                    ptr[c][d] = 4*((double)rand()/10000000);
                    if(p<0.0001)
                    {
                        ptr[c][d] = ptr[c][d] * (-1);
                    }
                }
            }
        }
        ~matrix()
        {
            for(int i=0;i<r;i++)
            {
                delete[] ptr[i];
            }
            delete[] ptr;
        }
        void display(void);
        void scale(double);
        void displaydims(void);
        friend void matmul(matrix *mat1,matrix *mat2, matrix *out);                     //done
        friend void mattrans(matrix *mat, matrix *out);                                 //done
        friend void matderiv(matrix *mat, matrix *out, int activ);                      //done
        friend void matactivate(matrix *mat,matrix *out, int activ);                    //done
        friend double costfunction(matrix *y, matrix *yhat, matrix *w, double lamda);   //done
        friend void matbroadcast(matrix *matout, matrix *matadd);                       //done
        friend void dzlast(matrix *y, matrix *yhat, matrix *dz);                        //done
        friend void finddb(matrix *dz, matrix *db, int m);                              //done
        friend void finddw(matrix *dz, matrix *anlast, matrix *dw, int m);              //done
        friend void change(matrix *w, matrix *dw, double alpha, double lamda, int m);   //done
        friend void dzothers(matrix*, matrix*, matrix*, matrix*);                       //done
        friend void copymat(matrix *mat, matrix *out, int from, int to);                //done
        friend void findconfusion(matrix *mat, matrix *out, matrix *conf);              //done
        friend void dropout(matrix *mat, double p);                                     //done
};

//Displays the matrix, defined as class matrix's function
void matrix::display(void)
{
    int c,d;
    for(c=0;c<this->r;c++)
    {
        for(d=0;d<this->c;d++)
        {
            cout<<ptr[c][d]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

void matrix :: scale(double s)
{
    int c,d;
    for(c=0;c<this->r;c++)
    {
        for(d=0;d<this->c;d++)
        {
            ptr[c][d] *=s;
        }
    }
}

void matrix::displaydims(void)
{
    cout<<"("<<r<<", "<<c<<")"<<endl;
}

//provides the multiplication of mat1 and mat 2 and stores it in out
void matmul(matrix *mat1, matrix *mat2, matrix *out)
{
    int c,d,k;
    double sum=0;
    for(c=0;c<mat1->r;c++)
    {
        for(d=0;d<mat2->c;d++)
        {
            for(k=0;k<mat2->r;k++)
            {
                sum = sum + mat1->ptr[c][k] * mat2->ptr[k][d];
            }
            out->ptr[c][d] = sum;
            sum = 0;
        }
    }
}
//gives the transpose of a matrix and stores it in the matrix pointed by 'out'
void mattrans(matrix *mat, matrix *out)
{
    int c,d;
    for (c=0;c<mat->r;c++)
    {
        for(d=0;d<mat->c;d++)
        {
            out->ptr[d][c] = mat->ptr[c][d];
        }
    }
}

//calculates activations based on mat and activation function
void matactivate(matrix *mat,matrix *out, int activ)
{
    if(activ == 0)//ReLu
    {
        int c,d;
        for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                if(mat->ptr[c][d] < 0)
                {
                    out->ptr[c][d] = out->ptr[c][d] * 0.01;
                }
                else
                {
                    out->ptr[c][d] = mat->ptr[c][d];
                }
            }
        }
    }
    else if(activ==1)//Sigmoid
    {
        int c,d;
        for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                out->ptr[c][d] = pow(e,mat->ptr[c][d])/(1 + pow(e,mat->ptr[c][d]));
            }
        }
    }
    else if(activ==2)//Tanh
    {
        int c,d;
        for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                out->ptr[c][d] = (pow(e,2*mat->ptr[c][d]) - 1)/(1 + pow(e,2 * mat->ptr[c][d]));
            }
        }
    }
    else//activ==3, Softmax
    {
        int c,d;
        double *sum =  new double[out->c];
        //copy values in mat to out
        for(c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                out->ptr[c][d] = mat->ptr[c][d];
            }
        }
        //raise e to every value
        for(c=0;c<out->r;c++)
        {
            for(d=0;d<out->c;d++)
            {
                if(out->ptr[c][d] >= 0)
                {
                    out->ptr[c][d] = pow(e, out->ptr[c][d]);
                }
                else if(out->ptr[c][d] < 0)
                {
                    out->ptr[c][d] = pow((1/e),-1*(out->ptr[c][d]));
                }
            }
        }
        //vertical sum of all values
        for(d=0;d<out->c;d++)
        {
            sum[d] = 0;
            for(c=0;c<out->r;c++)
            {
                sum[d] += out->ptr[c][d];
            }
        }
        //divide every value by sum
        for(d=0;d<out->c;d++)
        {
            for(c=0;c<out->r;c++)
            {
                out->ptr[c][d] = out->ptr[c][d] / sum[d];
            }
        }
    }
}

//Calculates the derivative of the matrix based on the activation function used, read comment section at the start
void matderiv(matrix *mat, matrix *out, int activ)
{
    int c,d;
    if(activ==0)// activ ==0 : ReLu
    {
       for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                if(mat->ptr[c][d] < 0)
                    {out->ptr[c][d] = 0.01;}
                else
                    {out->ptr[c][d] = 1;}
            }
        }
    }
    else if(activ==1)//activ==1 : Sigmoid
    {
        for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                out->ptr[c][d] = (1 - mat->ptr[c][d]) * mat->ptr[c][d];
            }
        }
    }
    else//activ==2 : tanh
    {
        for (c=0;c<mat->r;c++)
        {
            for(d=0;d<mat->c;d++)
            {
                out->ptr[c][d] = 1 - pow(mat->ptr[c][d],2);
            }
        }
    }


}

//calculate cost function from actual output(y) and predicted output(yhat)
double costfunction(matrix *y, matrix *yhat, matrix *w, double lamda)
{
    //cost = (1/(m* number of classes)) * sum(y * log10(y_hat) + (1-y) * log10(1-y_hat)) + (lambda/2m) * (elementwise square sum W)
    //Step 1: find sum(y*log10(yhat) + (1-y)*log10(1-y_hat)) = out
    //Step 2: find sum-squared W = w2
    //Step 3: (1/(m*n)) * ((lamda * w2) + out)
    double out=0, w2=0,y1,yhat1;
    double cost = 0;
    int c,d;

    for(c=0;c<y->r;c++)
    {
        for(d=0;d<y->c;d++)
        {
            y1 = y->ptr[c][d];
            yhat1 = yhat->ptr[c][d];
            out = out - y1*log10(yhat1) - (1-y1)*log10(1-yhat1);
        }
    }
    for(c=0;c<w->r;c++)
    {
        for(d=0;d<w->c;d++)
        {
            w2 = w2 + pow(w->ptr[c][d],2);
        }
    }

    cost = (lamda * w2) + out;
    w2 = 1/(double(y->r) * double(y->c));
    cost = (cost) * w2;
    return(cost);
}

//broadcast and add constant matrix 'b' to 'z' matrix
void matbroadcast(matrix *matout, matrix *matadd)
{
    //matout is the z matrix and matadd is the 'b' or constant matrix
    int c,d;
    for(c=0;c<matout->r;c++)
    {
        for(d=0;d<matout->c;d++)
        {
            matout->ptr[c][d] = matout->ptr[c][d] + matadd->ptr[c][0];
        }
    }
}

//find the dz for last layer of the network
void dzlast(matrix *y, matrix *yhat, matrix *dz)
{
    int c,d;
    for(c=0;c<y->r;c++)
    {
        for(d=0;d<y->c;d++)
        {
            dz->ptr[c][d] = y->ptr[c][d] - yhat->ptr[c][d];
        }
    }
}

//find db from dz and size of training set
void finddb(matrix *dz, matrix *db, int m)
{
    //dbn = (1/m) * horizontal sum(dzn);
    int c,d;
    for(c=0;c<dz->r;c++)
    {
        db->ptr[c][0] = 0;
        for(d=0;d<dz->c;d++)
        {
            db->ptr[c][0] = db->ptr[c][0] + dz->ptr[c][d];
        }
    }
    db->scale(1/double(m));
}

//find dw from dz and a[n-1], scale by (1,m)
void finddw(matrix *dz, matrix *anlast, matrix *dw, int m)
{
    //dwn = (1/m) * dzn * a(n-1).T
    matrix *p;
    p = new matrix(anlast->c,anlast->r);
    mattrans(anlast, p);
    matmul(dz, p, dw);
    dw->scale(1/double(m));
    delete p;
}

//find w and b from original values and dw, db
void change(matrix *w, matrix *dw, double alpha, double lamda, int m)
{
    //wn = wn - alpha * dwn - (lamba/m) * wn
    matrix *p = new matrix(w);
    int c,d;
    for(c=0;c< w->r;c++)
    {
        for(d=0;d<w->c;d++)
        {
            p->ptr[c][d] = w->ptr[c][d];
        }
    }
    p->scale(1-((alpha*lamda)/double(m)));
    dw->scale(alpha);

    for(c=0;c< w->r;c++)
    {
        for(d=0;d<w->c;d++)
        {
            w->ptr[c][d] = p->ptr[c][d] - dw->ptr[c][d];
        }
    }

    delete p;
}

//dz for all other layers
void dzothers(matrix *wn, matrix *dzn, matrix *derz, matrix *dz)
{
    //For other layers, dzn = [w(n+1).T * dz(n+1)] * elementwise of derzn
    matrix *p=new matrix(wn->c,wn->r);
    //cout<<wn->r<<" = "<<p->r<<endl;
    //cout<<wn->c<<" = "<<p->c<<endl;
    mattrans(wn, p);
    matmul(p, dzn, dz);
    int c,d;
    for(c=0;c<dz->r;c++)
    {
        for(d=0;d<dz->c;d++)
        {
            dz->ptr[c][d] = dz->ptr[c][d] * derz->ptr[c][d];
        }
    }
    delete p;
}

void copymat(matrix *mat, matrix *out, int from, int to)
{
    int c,d;
        for(d=0;d<to-from;d++)
        {
            for(c=0;c<mat->r;c++)
            {
                out->ptr[c][d] = mat->ptr[c][d+from];
            }
        }
}

//Find True positives, True Negatives, False Positives and False Negatives
void findconfusion(matrix *mat, matrix *out, matrix *conf)
{
    int c,d; double maximum;
    matrix *counter;
    counter = new matrix (mat->r, mat->c);
    for(c=0;c<mat->c;c++)
    {
        //find the maximum of a column
        maximum = 0;
        for(d=0;d<mat->r;d++)
        {
            if(mat->ptr[d][c] > maximum)
            {
                maximum = mat->ptr[d][c];
            }
        }
        //classify using maximum acquired from above loop
        for(d=0;d<mat->r;d++)
        {
            if(mat->ptr[d][c] < maximum)//not equal to maximum i.e. zero predicted
            {
                if(out->ptr[d][c]==0)//predicted zero, is zero: True Negative
                {
                    conf->ptr[1][1]++;
                }
                else//predicted zero, is one: False Negative
                {
                    conf->ptr[0][1]++;
                }
            }
            else//equal to maximum probability i.e. one predicted
            {
                if(out->ptr[d][c]==0)//predicted one, is zero: False Positive
                {
                    conf->ptr[1][0]++;
                }
                else//predicted one, is one: True Positive
                {
                    conf->ptr[0][0]++;
                }
            }
        }
    }
    delete counter;
}

//Implement neural dropouts on random nodes based on probability 'p'
void dropout(matrix *mat, double p)
{
    int drops,i,j;
    drops = int(mat->r * p);
    double *ptr, minimum=1;
    ptr = new double[mat->r];
    //randomise the double array
    for(i=0;i<mat->r;i++)
    {
        ptr[i] = (double)rand()/100000;
    }
    for(i=0;i<drops;i++)//for number of neurons to be dropped,
    {
        minimum = 1;
        for(j=0;j<mat->r;j++)//find minimum in 'drop-th' iteration
        {
            if(ptr[i]<minimum && ptr[i]!=0)//value is minimum if it is non zero(dropped neurons will be zero after 1st iteration) and if it is lesser than other non zero values
            {
                minimum = ptr[i];
            }
        }

        for(j=0;j<mat->r;j++)//set minimum valued neuron to zero
        {
            if(ptr[i]==minimum)
            {
                ptr[i] = 0;
            }
        }
    }//dropped neurons created

    for(i=0;i<mat->r;i++)
    {
        if(ptr[i]==0)//if dropped neuron index is present, drop all values in ith row of all columns in the matrix 'mat'
        {
            for(j=0;j<mat->c;j++)//increase column indices of 'mat'
            {
                mat->ptr[i][j] = 0;
            }
        }
    }//neurons dropped in 'mat'
    delete ptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class layer
{
    public:
        int rl;
        int cl;
        int m;
        int activ;// 0: Relu, 1: Sigmoid, 2: Tanh, 3: Softmax
        matrix w;
        matrix dw;
        matrix b;
        matrix db;
        matrix z;
        matrix dz;
        matrix a;
        matrix derz;
        void operator = (int *setvals)       //constructor for the layer
        {
            int dims[2];
            this->rl = setvals[0], this->cl = setvals[1], this->m = setvals[2], this->activ = setvals[3];
            dims[0] = rl;
            dims[1] = cl;
            w = (dims);
            dw = (dims);
            dims[1] = 1;
            b = (dims);
            db = (dims);
            dims[1] = m;
            z = (dims);
            dz = (dims);
            a = (dims);
            derz = (dims);
        }

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class net
{
    public:
        layer *layerptr;
        matrix train;
        matrix cross;
        matrix trainout;
        matrix crossout;
        matrix *input;
        matrix *output;
        int mini;// mini batch size
        int nx;//number of features
        int classes;//number of classes
        int t;//total samples / mini batch size
        int nl;//number of layers
        int epoch;//number of training iterations
        int activ;
        int trainsize;
        double lamda;//regularisation parameter
        double alpha;
        double droprate;//stores the cost of the network in each iteration
        net(matrix *ip, matrix *op, int trainsz,int nlayer = 5, int hidden=10, int act=0, int m=1000, int ep = 5000, double al = 0.001, double lam = 0.01, double dr=0.5)
        {
            input = ip, output = op;
            nx = input->r;
            classes = op->r;
            mini = m;
            t = input->c / m;
            nl = nlayer;
            epoch = ep;
            lamda = lam;
            activ = act;
            nl = nlayer;
            trainsize = trainsz;
            alpha = al;
            droprate = dr;
            if(input->c != output->c)
            {
                cout<<"Error : Number of samples in Input and Output Matrices do not match!!"<<endl;exit(0);
            }
            /*if(hidden<classes)
            {
                cout<<"Number of hidden units have been increased for processing."<<endl;
                hidden = classes;
            }
            if(hidden>classes)
            {
                cout<<"Number of hidden units have been decreased for processing."<<endl;
                hidden = classes;
            }*/
            /*
            nlayer : Number of Layers
            hidden : Number of hidden units per layer
            activ : Activation Function:
                                    0: ReLu
                                    1: Sigmoid
                                    2: Tanh
                                    3: Softmax (Last Layer)
            m: Mini Batch Size : default size 1000
            */
            int setvals[4];

            layerptr = new layer[nlayer];
            setvals[0] = hidden;
            setvals[1] = nx;
            setvals[2] = m;
            setvals[3] = activ;
            layerptr[0]=(setvals);
            setvals[1] = hidden;
            int k;
            for (k=1;k<nlayer-1;k++)
            {
                layerptr[k] = (setvals);
            }
            setvals[0] = classes;
            setvals[3] = 3;
            layerptr[nlayer-1] = (setvals);
            cout<<"Mini Batch Size = "<<m<<endl;
            cout<<"Total Number of Minibatches = "<<t<<endl;
            cout<<"Number of Mini Batches in training = "<<trainsize/mini<<endl;
            cout<<"Number of Mini Batches in crossvalidation = "<<t-(trainsize/mini)<<endl;
            cout<<"Number of classes = "<<classes<<endl;
            cout<<"Activation Function  = "<<activ<<endl;
            cout<<"Epoch Size = "<<epoch<<endl;
            cout<<"Learning Rate = "<<alpha<<endl;
            cout<<"Regularisation Parameter : "<<lamda<<"\n"<<endl;
            cout<<"\nLayers:"<<endl;
            cout<<input->r<<"\t";
            for (k=0;k<nl;k++)
            {
                cout<<layerptr[k].rl<<"\t";
            }
            cout<<endl;
            int dims[2];

            dims[0] = nx;
            dims[1] = trainsize;
            train = (dims);
            dims[1] = input->c - trainsize;
            cross = (dims);
            dims[0] = classes;
            dims[1] = trainsize;
            trainout = (dims);
            dims[1] = output->c - trainsize;
            crossout = (dims);
        }
        void trainnet(void);
        double forwardprop(matrix *matin, matrix *matout,double drop);//calculate forward propagation of along the network for minibatch size
        void backprop(matrix *, matrix*);// calculate backprop for all the layers
        void test(matrix *testin, matrix *testout); // to use forward propagation on test set and calculate precision, recall and F1 Score
};

//forward propagation function
double net::forwardprop(matrix *matin, matrix *matout, double drop)
{
    double c;
    int i=0;
    matmul(&layerptr[0].w, matin, &layerptr[0].z);
    matbroadcast(&layerptr[0].z, &layerptr[0].b);
    dropout(&layerptr[0].z, drop);
    matactivate(&layerptr[0].z, &layerptr[0].a, layerptr[0].activ);

    for(i=1;i<nl;i++)
    {
        matmul(&layerptr[i].w, &layerptr[i-1].a, &layerptr[i].z);
        matbroadcast(&layerptr[i].z, &layerptr[i].b);
        dropout(&layerptr[i].z, drop*(1+(i/(nl*nl))));
        matactivate(&layerptr[i].z, &layerptr[i].a, layerptr[i].activ);
    }
    c = costfunction(matout, &layerptr[nl-1].a, &layerptr[nl-1].w, lamda);
    return(c);
}

//backward propagation function
void net::backprop(matrix *matin, matrix *matout)
{
    /*Back Propagation
    For last layer, dzlast = y - y_pred
    For other layers, dzn = [w(n+1).T * dz(n+1)] * elementwise of derzn

    derivatives of zn =>
    sigmoid derivative = (1-g(z)) * g(z)
    tanh derivative = 1 - g(z)^2
    relu derivative = 1, for all z>=0
                    = 0, otherwise

    dwn = (1/m) * dzn * a(n-1).T
    dbn = (1/m) * horizontal sum(dzn)

    wn = wn - alpha * dwn - (lamba/m) * wn
    bn = bn - alpha * dbn - (lamba/m) * bn

    stages:
    1. Find dz of current layer
    2. Find dw and db of current layer
    3. use dw and db to find w and b of current layer*/
    int i;

    for(i=1;i<nl;i++)
    {
        if(i==1)//different backprop parts of last layer
        {
            dzlast(matout, &layerptr[nl-i].a, &layerptr[nl-i].dz);//dzlast = y - yhat
            finddw(&layerptr[nl-i].dz, &layerptr[nl-(i+1)].a, &layerptr[nl-i].dw, trainsize);//dwlast = (1/m)*[dz * a(n-1).T]
        }
        else //for all layers other than last layer
        {
            dzothers(&layerptr[nl-(i-1)].w, &layerptr[nl-(i-1)].dz, &layerptr[nl-i].derz, &layerptr[nl-i].dz);//dz = [w(n+1).T * dz(n+1)] * derz
            if(i==nl)//special dw case for first layer as it requires input matrix 'matin'
            {
                finddw(&layerptr[nl-i].dz, matin, &layerptr[nl-i].dw, trainsize);//dw = (1/m) * [dz * matin.T]
            }
            else//dw for all other layers
            {
                finddw(&layerptr[nl-i].dz, &layerptr[nl-(i+1)].a, &layerptr[nl-i].dw, trainsize);//dw = (1/m) * [dz * a(n-1).T]
            }
        }
        //common functions to find db and change the values of w and b by regularisation
        finddb(&layerptr[nl-i].dz, &layerptr[nl-i].db, trainsize); //dm = (1/m) * horizontalsum(dz)
        change(&layerptr[nl-i].w, &layerptr[nl-i].dw, alpha, lamda, trainsize);//w = w - alpha*dw - (lamda/m).*w
        change(&layerptr[nl-i].b, &layerptr[nl-i].db, alpha, lamda, trainsize);//b = b - alpha*db - (lamda/m).*b
    }
}

void net::trainnet(void)
{
    int from=0,to;
    to = from+mini-1;
    double coste[this->epoch];
    double costc[this->epoch];
    matrix miniin(train.r, this->mini);//minibatch sized matrix to forwardprop and backprop
    matrix miniout(trainout.r,this->mini);//minibatch sized output matrix to forwardprop and backprop
    int c=0,d=0,k=0;
    int ntbatch = trainsize/mini;// number of minibatches in training
    int ncbatch = t-ntbatch;//number of minibatches in crossvalidation
    cout<<"\nTRAINING THE DATASET!: Check Cost for Training Data and Crossvalidation Data : "<<endl;
    cout<<"-------------------------------------------------------------------------------------"<<endl;
    for(c=0;c<this->epoch;c++)
    {
        coste[c] = 0;
        for(d=0;d<ntbatch;d++)
        {
            from = d*mini;
            to = from + mini - 1;
            copymat(input,&miniin,from,to);
            copymat(output,&miniout,from,to);
            coste[c] = forwardprop(&miniin, &miniout, droprate) + coste[c];
            backprop(&miniin, &miniout);
        }
        coste[c] = coste[c]/double(mini*ntbatch);
        for(k=0;k<ncbatch;k++)
        {
            from = k*mini + ntbatch*mini;
            to = from + mini - 1;
            copymat(input,&miniin,from,to);
            copymat(output,&miniout,from,to);
            costc[c] = forwardprop(&miniin, &miniout,0) + coste[c];
        }
        costc[c] = costc[c]/double(mini*ncbatch);
        if(c%100==0)
        cout<<"Costtrain["<<c<<"] = "<<coste[c]<<"; Costcross["<<c<<"] = "<<costc[c]<<endl;
    }
cout<<"-------------------------------------------------------------------------------------"<<endl;
}

//function to calculate precision, recall and F1 score from Test Set
void net::test(matrix *testin, matrix *testout)
{
    cout<<"\nOutput on Test Data is : "<<endl;
    int from=0,to;
    to = from+mini-1;
    matrix miniin(train.r, this->mini);//minibatch sized matrix to forwardprop
    matrix miniout(trainout.r,this->mini);//minibatch sized output matrix to forwardprop
    matrix confusion(2,2);//calculate True positives, False positives, True Negative, False Negative
    confusion.ptr[0][0]=0;confusion.ptr[0][1]=0;confusion.ptr[1][0]=0;confusion.ptr[1][1]=0;
    int c=0,d=0,k=0;
    int ntbatch = testin->c/mini;// number of minibatches in testing
    double costfinal=0;
    for(d=0;d<ntbatch;d++)
        {
            from = d*mini;
            to = from + mini - 1;
            copymat(input,&miniin,from,to);
            copymat(output,&miniout,from,to);
            costfinal = forwardprop(&miniin, &miniout,0) +costfinal;
            findconfusion(&layerptr[this->nl-1].a, &miniout, &confusion);
        }
    costfinal = costfinal/double(testin->c);
    double precision, recall, f1;
    precision = (confusion.ptr[0][0]/(confusion.ptr[0][0]+confusion.ptr[1][0]));
    recall = (confusion.ptr[0][0]/(confusion.ptr[0][0]+confusion.ptr[0][1]));
    f1 = (2*precision*recall)/(precision + recall);
    precision*=100;
    recall*=100;
    f1*=100;
    cout<<"Precision : "<<precision<<endl;
    cout<<"Recall : "<<recall<<endl;
    cout<<"F1 Score : "<<f1<<endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function for aid in randomising Indices
int check(int *a, int d)
{
    int c,flag=0;
    for(c=0;c<270;c++)
    {
        if(a[c]==d)
        {
            return(flag);
        }
    }
    flag=1;
    return(flag);
}
