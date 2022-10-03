#include <stdio.h>
#include<conio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
    //Variable declaration
    int L;      //No of inputs
    int M = 3;  //No of hidden neurons
    int N;      //No of outputs
    int P;      //No of training patterns
    int T;      //No of testing patterns


    int i,j,k,p;
    int iteration = 1;

    float TMSE = 100;
    float aTMSE;
    float LR = 0.5; //Learning rate

    FILE *input;
    FILE *toutput;
    FILE *ainput;
    FILE *atoutput;
    FILE *output1;
    FILE *output2;
    FILE *output3;

    //Taking inputs from the user
    printf("Enter the Number of inputs\n");
    scanf("%d",&L);
    printf("Enter the Number of outputs\n");
    scanf("%d",&N);
    printf("Enter the Number of training patterns\n");
    scanf("%d",&P);
    printf("Enter the Number of testing patterns\n");
    scanf("%d",&T);

    //Reading the input file
    float I[P+1][L+1];
    float Itemp[P+1][L+1];

    input = fopen("input.txt","r");

    for(p=1;p<=P;p++)
    {
        for(i=1;i<=L;i++)
        {
            fscanf(input,"%f",&I[p][i]);
        }
    }

    fclose(input);

   //Normalizing the input
    float max, min;

    for(p=1;p<=P;p++)
    {
        for(i=1;i<=L;i++)
        {
            Itemp[p][i] = I[p][i];
        }
    }

    printf("\n");

    for(i=1;i<=L;i++)
    {
        max = I[1][i];
        min = I[1][i];

        for(p=1;p<=P;p++)
        {
            if(max<=I[p][i])
            {
                max = I[p][i];
            }

            if(min>=I[p][i])
            {
                min = I[p][i];
            }
        }

        for(p=1;p<=P;p++)
        {
            I[p][i] = (((Itemp[p][i]-min)*0.8)/(max-min)) + 0.1;
        }
    }

    //Reading Target output
        float TO[P+1][N+1];
        float TOtemp[P+1][N+1];

        toutput = fopen("toutput.txt","r");

        for(p=1;p<=P;p++)
        {
            for(k=1;k<=N;k++)
            {
                fscanf(toutput,"%f",&TO[p][k]);
            }
        }

        fclose(toutput);

    //Normalizing target output

        printf("\n");

        for(p=1;p<=P;p++)
        {
            for(k=1;k<=N;k++)
            {
                TOtemp[p][k] = TO[p][k];
            }
        }

        for(k=1;k<=N;k++)
        {
            max = TO[1][k];
            min = TO[1][k];

            for(p=1;p<=P;p++)
            {
                if(max<=TO[p][k])
                {
                    max = TO[p][k];
                }
                if(min>=TO[p][k])
                {
                    min = TO[p][k];
                }
            }

            for(p=1;p<=P;p++)
            {
                TO[p][k] = (((TOtemp[p][k]-min)*0.8)/(max-min)) + 0.1;
            }
        }

    //Bias1
    for(p=1;p<=P;p++)
    {
        I[p][0] = 1.0;
    }

    //Weights initialization
    float V[L+1][M+1];
    float W[M+1][N+1];

    for(i=0;i<=L;i++)
    {
        for(j=1;j<=M;j++)
        {
            V[i][j] = (float)(rand()%10)/(float)10;
        }
    }

    for(j=0;j<=M;j++)
    {
        for(k=1;k<=N;k++)
        {
            W[j][k] = (float)(rand()%10)/(float)10;
        }
    }

    output1 = fopen("output1.txt","w");
    fprintf(output1,"iteration\tTMSE\n");

    output3 = fopen("output3.txt","w");

    while(TMSE>0.001 && iteration<=100000)
    {
        //Input to the hidden layer
        float IH[P+1][M+1];

        for(p=1;p<=P;p++)
        {
            for(j=1;j<=M;j++)
            {
                IH[p][j] = 0;
                for(i=0;i<=L;i++)
                {
                    IH[p][j] = IH[p][j] + (I[p][i] * V[i][j]);
                }
            }
        }

        //Output of the hidden layer (TF Log-sigmoid)
        float OH[P+1][M+1];

        for(p=1;p<=P;p++)
        {
            for(j=1;j<=M;j++)
            {
                OH[p][j] = 1.0/(1.0+exp(-1.0*I[p][j]));
            }
        }

        //Bias2
        for(p=1;p<=P;p++)
        {
            OH[p][0] = 1.0;
        }

        //Input to output layer
        float IO[P+1][N+1];

        for(p=1;p<=P;p++)
        {
            for(k=1;k<=N;k++)
            {
                IO[p][k] = 0;
                for(j=0;j<=M;j++)
                {
                    IO[p][k] = IO[p][k] + (OH[p][j] * W[j][k]);
                }
            }
        }

        //Output of the output layer (TF Tan-sigmoid)
        float OO[P+1][N+1];

        for(p=1;p<=P;p++)
        {
            for(k=1;k<=N;k++)
            {
                OO[p][k] = 1.0/(1.0+exp(-1.0*IO[p][k]));
            }
        }

        //MSE calculation
        float MSE[P+1][N+1];
        TMSE = 0;
        for(p=1;p<=P;p++)
        {
            for(k=1;k<=N;k++)
            {
                MSE[p][k] = (0.5 * (TO[p][k]-OO[p][k]) * (TO[p][k]-OO[p][k]));

                TMSE = TMSE + MSE[p][k];

            }
        }
        TMSE = TMSE/P;

        fprintf(output1,"%d\t\t%f\n",iteration,TMSE);
        printf("%d\t\t%f\n",iteration,TMSE);

        //Updating weights
        float DV[L+1][M+1];
        float DW[M+1][N+1];

        for(j=0;j<=M;j++)
        {
            for(k=1;k<=N;k++)
            {
                DW[j][k] = 0.0;
                for(p=1;p<=P;p++)
                {
                    DW[j][k] = DW[j][k] + ((TO[p][k]-OO[p][k])*OO[p][k]*(1-OO[p][k])*OH[p][j]);
                }
                DW[j][k] = (LR*DW[j][k])/((float)P);
            }
        }

        for(i=0;i<=L;i++)
        {
            for(j=1;j<=M;j++)
            {
                DV[i][j] = 0.0;
                for(p=1;p<=P;p++)
                {
                    for(k=1;k<=N;k++)
                    {
                        DV[i][j] = DV[i][j] + ((TO[p][k]-OO[p][k])*OO[p][k]*(1-OO[p][k])*W[j][k]*I[p][i]*OH[p][j]*(1-OH[p][j]));
                    }
                }
                DV[i][j] = (LR*DV[i][j])/((float)(P*N));
            }
        }

        for(j=0;j<=M;j++)
        {
            for(k=1;k<=N;k++)
            {
                W[j][k] = W[j][k] + DW[j][k];
            }
        }

        for(i=0;i<=L;i++)
        {
            for(j=1;j<=M;j++)
            {
                V[i][j] = V[i][j] + DV[i][j];
            }
        }

        iteration = iteration + 1;
    }

    fprintf(output3,"\nFor the training number of iterations required = %d\nand the average mean square error is  = %f\n",iteration-1,TMSE);

    fclose(output1);

    fprintf(output3,"\n\nV values:\n");

    for(i=0;i<=L;i++)
    {
        for(j=1;j<=M;j++)
        {
            fprintf(output3,"%f\t\t",V[i][j]);
        }
        fprintf(output3,"\n");
    }

    fprintf(output3,"\nW values:\n");

    for(j=0;j<=M;j++)
    {
        for(k=1;k<=N;k++)
        {
            fprintf(output3,"%f\t\t",W[j][k]);
        }
        fprintf(output3,"\n");
    }

    //Testing:===================================================================================================================

    //Reading the testing input file
    float aI[T+1][L+1];
    float aItemp[T+1][L+1];

    ainput = fopen("ainput.txt","r");

    for(p=1;p<=T;p++)
    {
        for(i=1;i<=L;i++)
        {
            fscanf(ainput,"%f",&aI[p][i]);
        }
    }

    fclose(ainput);

    //Normalizing the testing input

    for(p=1;p<=T;p++)
    {
        for(i=1;i<=L;i++)
        {
            aItemp[p][i] = aI[p][i];
        }
    }

    for(i=1;i<=L;i++)
    {
        max = aI[1][i];
        min = aI[1][i];

        for(p=1;p<=T;p++)
        {
            if(max<=aI[p][i])
            {
                max = aI[p][i];
            }

            if(min>=aI[p][i])
            {
                min = aI[p][i];
            }
        }

        for(p=1;p<=T;p++)
        {
            aI[p][i] = (((aItemp[p][i]-min)*0.8)/(max-min)) + 0.1;
        }
    }

        for(p=1;p<=T;p++)
        {
            for(k=1;k<=N;k++)
            {
                printf("\n%f\n",aI[p][k]);
            }
        }


    //Reading Target output
        float aTO[T+1][N+1];
        float aTOtemp[T+1][N+1];

        atoutput = fopen("atoutput.txt","r");

        for(p=1;p<=T;p++)
        {
            for(k=1;k<=N;k++)
            {
                fscanf(atoutput,"%f",&aTO[p][k]);
            }
        }

        fclose(atoutput);

    //Normalizing target output

        for(p=1;p<=T;p++)
        {
            for(k=1;k<=N;k++)
            {
                aTOtemp[p][k] = aTO[p][k];
            }
        }

        for(k=1;k<=N;k++)
        {
            max = aTO[1][k];
            min = aTO[1][k];

            for(p=1;p<=T;p++)
            {
                if(max<=aTO[p][k])
                {
                    max = aTO[p][k];
                }
                if(min>=aTO[p][k])
                {
                    min = aTO[p][k];
                }
            }

            for(p=1;p<=T;p++)
            {
                aTO[p][k] = (((aTOtemp[p][k]-min)*0.8)/(max-min)) + 0.1;
            }
        }

        for(p=1;p<=T;p++)
        {
            for(k=1;k<=N;k++)
            {
                printf("\n%f\n",aTO[p][k]);
            }
        }


    //Input to the hidden layer
    float aIH[T+1][M+1];

    for(p=1;p<=T;p++)
    {
        for(j=1;j<=M;j++)
        {
            aIH[p][j] = 0.0;
            for(i=0;i<=L;i++)
            {
                aIH[p][j] = aIH[p][j] + (aI[p][i] * V[i][j]);
            }
        }
    }

    //Output of the hidden layer (TF Log-sigmoid)
    float aOH[T+1][M+1];

    for(p=1;p<=T;p++)
    {
        for(j=1;j<=M;j++)
        {
            aOH[p][j] = 1.0/(1.0+exp(-1.0*aI[p][j]));
        }

        if(p==1)
            {
                printf("\n%f\n",aOH[p][j]);
            }
    }

    //Bias2

    for(p=1;p<=T;p++)
    {
        aOH[p][0] = 1.0;
    }

    //Input to output layer
    float aIO[T+1][N+1];

    for(p=1;p<=T;p++)
    {
        for(k=1;k<=N;k++)
        {
            aIO[p][k] = 0.0;
            for(j=0;j<=M;j++)
            {
                aIO[p][k] = aIO[p][k] + (aOH[p][j]*W[j][k]);
            }
            if(p==1)
            {
                printf("\n%f\n",aIO[p][k]);
            }
        }
    }

    //Output of the output layer (TF Tan-sigmoid)
    float aOO[T+1][N+1];

    for(p=1;p<=T;p++)
    {
        for(k=1;k<=N;k++)
        {
            aOO[p][k] = 1.0/(1.0+exp(-1.0*aIO[p][k]));

            if(p==1)
            {
                printf("\n%f\n",aOO[p][k]);
            }
        }
    }

    output2 = fopen("output2.txt","w");

    fprintf(output2,"i = iteration\nTO = target output\nOO = obtained output\n\n");
    fprintf(output2,"\tTO\t\t  OO\n");

    for(p=1;p<=T;p++)
    {
        for(k=1;k<=N;k++)
        {
            fprintf(output2,"\t%f\t%f",aTO[p][k],aOO[p][k]);
        }
        fprintf(output2,"\n");
    }

    fclose(output2);

    //MSE calculation
    float aMSE[T+1][N+1];

    for(p=1;p<=T;p++)
    {
        for(k=1;k<=N;k++)
        {
            aMSE[p][k] = (0.5 * (aTO[p][k]-aOO[p][k]) * (aTO[p][k]-aOO[p][k]));

            aTMSE = aTMSE + aMSE[p][k];
        }
    }

    aTMSE = aTMSE/((float)(N*T));

    fprintf(output3,"\n\nThe MSE for 'testing' is %f\n",aTMSE);
    printf("\n\nThe MSE for 'testing' is %f\n",aTMSE);

    fclose(output3);

    return 0;
}
