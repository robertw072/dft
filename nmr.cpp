#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

using namespace std;

// A struct to hold xy points read from data file
struct xy_point 
{
	double x;
	double y;
};

// Function to help with sorting
bool comparex(xy_point x1, xy_point x2)
{
	if (x1.x < x2.x)
		return true;

	return false;
}

// Boxcar filter function
void boxcar(vector<xy_point> &filter, int n)
{
	double coeff = 1.0/double(n);
	int f = filter.size();

	int i = 0;
	while (i < f)
	{
		double sumY = 0.0;
		int j = ((i-((n-1)/2)%(f)+(f))%(f))-1;
		for (int k = 0; k < n; k++)
		{
			sumY = sumY + filter[j].y;
			j = ((j+1)%(f)+(f))%(f);
			//cout << j << " " << filter[j].y << endl;
		}
		sumY = sumY / n;
		filter[i].y = sumY;
		i++;	
	}

}

void sg(vector<xy_point> &filter, int n)
{
	// n is 5, 11, 17
	int f = filter.size();
	int constants5[] = {-3, 12, 17, 12, -3};
	int constants11[] = {-36, 9, 44, 69, 84, 89, 84, 69, 44, 9};
	int constants17[] = {-21, -6, 7, 18, 27, 34, 39, 42, 43, 42, 29, 34, 27, 18, 7, -6, -21};

	int i = 0;
	while (i < f)
	{
		double sumY = 0.0;
		int j = ((i-((n-1)/2)%(f)+(f))%(f))-1;
		for (int k = 0; k < n; k++)
		{	
			if (n == 5)
				sumY = sumY + ((constants5[k]*filter[j].y) / 35.0);
			else if (n == 11)
				sumY = sumY + ((constants11[k]*filter[j].y) / 429.0);
			else
				sumY = sumY + ((constants17[k]*filter[j].y) / 323.0);
			
			j = ((j+1)%(f)+(f))%(f);
		}
		filter[i].y = sumY;
		i++;
	}
}

void dft(vector<double> dataY, int method, double tol)
{
	//cout << "im in the function" << endl;
    int n = dataY.size();
	gsl_matrix_complex *Z = gsl_matrix_complex_alloc(n, n);	// matrix Z
	gsl_vector_complex *y = gsl_vector_complex_alloc(dataY.size()); // vector to hold Y values
	gsl_complex complexY;

	for (int i = 0; i < dataY.size(); i++)
	{
		GSL_SET_COMPLEX(&complexY, dataY[i], 0.0);		// make y values into complex numbers 
		gsl_vector_complex_set(y, i, complexY);			// store complex y's in a vector 
	}

	//cout << "am i past the for loop? yuh" << endl;

	double x = ((-2*M_PI)/(double) n);
	gsl_complex eu;
	GSL_SET_COMPLEX(&eu, cos(x), sin(x));

	//cout << "im not that far into the function tho" << endl;

	// compute and store complex values in matrix Z
	for(int j = 0; j <= n-1; j++)
	{
		for(int k = 0; k <= n-1; k++)
		{
            for (int m = 0; m < j*k; m++)
			    eu = gsl_complex_mul(eu, eu);	// compute e^ix to the jk power

			eu = gsl_complex_div_real(eu, sqrt((double) n));	// compute eu / sqrt(n)
			gsl_matrix_complex_set(Z, j, k, eu);
           //printf ("m(%d,%d) = %g\n", j, k, gsl_matrix_complex_get (Z, j, k));
		}
	}
	cout << "im out the for loop" << endl;
	gsl_vector_complex *c = gsl_vector_complex_alloc(n);
	// compute values of c, change this to use gsl_blas_zgemv function 
    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, Z, y, GSL_COMPLEX_ZERO, c);

//	cout << "im farther into the function now" << endl;

    gsl_matrix_complex *d = gsl_matrix_complex_alloc(n, n); // matrix delta used to compute matrix g values
    gsl_complex complexD;
    // construct the identity matrix
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            if (j==k)
            {
                GSL_SET_COMPLEX(&complexD, 1.0, 0.0);   // set complex 1
                gsl_matrix_complex_set(d, j, k, complexD);
            }
            else
            {
                GSL_SET_COMPLEX(&complexD, 0.0, 0.0);   // set complex 0
                gsl_matrix_complex_set(d, j, k, complexD);
            }
            // printf ("m(%d,%d) = %g\n", j, k, gsl_matrix_complex_get (d, j, k));
        }
    }

    gsl_matrix_complex *G = gsl_matrix_complex_alloc(n, n); // matrix G to compute Fourier coefficients
    gsl_complex g;
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            double r = exp((-4*log(2)*j*k)/pow(n, (double) 3 / 2));
            g = gsl_complex_mul_real(gsl_matrix_complex_get(d, j, k), r);
            gsl_matrix_complex_set(G, j, k, g);
        }
    }

    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, G, c, GSL_COMPLEX_ZERO, c);   // multiply matrix G by vector c to get Fourier coefficients

//cout << "im here bruh" << endl;

    if (method == 0)                                                // solve y = Zi*c, if method == 0
    {
        gsl_matrix_complex *Zi = gsl_matrix_complex_alloc(n, n);    // matrix Zi is the inverse of Z, will hold complex conjugates of Z
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                gsl_complex comcon = gsl_complex_conjugate(gsl_matrix_complex_get(Z, j, k));    // get the complex conjugate 
                gsl_matrix_complex_set(Zi, j, k, comcon);                                       // store it in Zi
            }
        }

        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, Zi, c, GSL_COMPLEX_ZERO, y);              // multiply matrix Zi by c to get filtered y values
    }

    else if (method == 1)                                           // solve c = Z*y, direct
    {
        gsl_permutation *p = gsl_permutation_alloc(n);
        int signum = 0;
        gsl_linalg_complex_LU_decomp(Z, p, &signum);

        gsl_linalg_complex_LU_solve(Z, p, c, y);
    }

    else                                                            // solve c = Z*y, iterative 
    {
        int k = 0;
        int max_iter = 100000;
        while (k <= max_iter)
        {
            gsl_complex sum;
            for (int i = 0; i < n; i++)
            {    
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        sum = gsl_complex_mul(gsl_matrix_complex_get(Z, i, j), gsl_vector_complex_get(y, j));
                        sum = gsl_complex_add(sum, gsl_vector_complex_get(c, i));
                    }
                }
                sum = gsl_complex_div(sum, gsl_matrix_complex_get(Z, i, i));

                gsl_complex sum2 = gsl_complex_sub(sum, gsl_vector_complex_get(y, i));
                if (sqrt(pow(GSL_REAL(sum2), 2) + pow(GSL_IMAG(sum2), 2)) < tol)
                {
                    gsl_vector_complex_set(y, i, sum);
                }
                k = k + 1;
            }
           
        }
    }

     for (int i = 0; i < n; i++)
    {
        printf ("v(%d) = %g\n", i, GSL_REAL(gsl_vector_complex_get(y, i)));
        dataY[i] = GSL_REAL(gsl_vector_complex_get(y, i));
    }

}

// Function that represents the cubic spline
double fx(double x, double a[], double b[], double c[], double d[], vector<xy_point> xy)
{
	double res;
	for (int i = 0; i < xy.size(); i++)
	{
		res = a[i] + (b[i]*(x-xy[i].x)) + (c[i]*(pow(x-xy[i].x,2.0))) + (d[i]*(pow(x-xy[i].x,3.0)));
		if ( xy[i].x <= x && x <= xy[i+1].x )
			return res;
	}
	return 0.0;	
}

int main()
{
	ifstream inputfile ("nmr.in"); // open the nmr.in file with program settings
				      // The nmr.in file takes one line with each paramter seperated by a space
				      // Parameter 1: data file
				      // Parameter 2: baseline
				      // Parameter 3: tolerance for numerical methods
				      // Paramater 4: type of filter (boxcar, sg, dft0, dft1, dft2) (0=inverse, 1=direct, 2=iterative)
				      // Parameter 5: filter size (0 turns off filtering)
				      // Parameter 6: number of filter passes, won't be considered if dft is selected
				      // Parameter 7: integration technique (0: adaptive quadrature, 1: Romberg, 2: Trap Rule, 3: Gaussian)
	
	// Pass nmr.in settings into variables
	string datafile;
	double baseline;
	double tol;
	string filtertype;
	int filtersize;
	int filterpasses;
	int method;		      // 0: Adaptive Quadrature, 1: Romberg, 2: Trap Rule, 3: Gaussian quadrature

	inputfile >> datafile >> baseline >> tol >> filtertype >> filtersize >> filterpasses >> method;

	// Read the data file and store the points in a vector of xy_points
	ifstream file (datafile);
	
	char delim = ' ';
	vector<xy_point> data;
	xy_point tmp;
	while (file >> tmp.x >> tmp.y)
	{
		data.push_back(tmp);
	}

	// Sorts the data in order of ascending x values (since the data goes down originally)
	sort(data.begin(), data.end(), comparex);

	vector<double> initroots;
	for (int i = 0; i < data.size(); i++)
	{
		if ((data[i].y-baseline) * (data[i+1].y-baseline) < 0.0)
		{
			initroots.push_back((data[i].x + data[i+1].x) / 2.0);
		}
	}

	double tmspeak = (initroots[initroots.size()-1] + initroots[initroots.size()-2]) / 2.0;
//	cout << tmspeak << endl;
	cout << "im here man" << endl;
	for (int i = 0; i < data.size(); i++)
		data[i].x = data[i].x - tmspeak;

	// Apply boxcar filter	
	if (filtertype == "boxcar" && filtersize != 0)
	{
		for (int i = 0; i < filterpasses; i++)
			boxcar(data, filtersize);
	}

	// Apply SG filter
	if (filtertype == "sg" && filtersize != 0)
	{
		for (int i = 0; i < filterpasses; i++)
			sg(data, filtersize);
	}

	if (filtertype == "dft0")
	{
		vector<double> y;
		for (int i = 0; i < data.size(); i++)
			y.push_back(data[i].y);

		dft(y, 0, tol);
	}

	if (filtertype == "dft1")
	{
		vector<double> y;
		for (int i = 0; i < data.size(); i++)
			y.push_back(data[i].y);

		dft(y, 1, tol);
	}

	if (filtertype == "dft2")
	{
		vector<double> y;
		for (int i = 0; i < data.size(); i++)
			y.push_back(data[i].y);

		dft(y, 2, tol);
	}

	// Initialize vector to hold functions that make up the cubic spline
	double a[data.size()], b[data.size()], c[data.size()], d[data.size()];
	int n = data.size() - 1;
	
	// Cubic spline algorithm	
	// Store y values in a
	for (int i = 0; i <= n; i++)
	{
		a[i] = data[i].y;
	}

	// Step 1
	double h[n];
	for (int i = 0; i <= n - 1; i++)
	{
		h[i] = data[i+1].x - data[i].x; 
	}

	// Step 2
	double A[n];
	for (int i = 1; i <= n - 1; i++)
	{
		A[i] = ((3.0/h[i])*(a[i+1]-a[i]))-((3.0/h[i-1])*(a[i]-a[i-1]));
	} 	

	// Step 3
	double l[n], u[n], z[n];
	l[0] = 1.0;
	u[0] = 0.0;
	z[0] = 0.0;

	// Step 4
	for (int i = 1; i <= n - 1; i++)
	{
		l[i] = 2*(data[i+1].x - data[i-1].x) - (h[i-1]*u[i-1]);
		u[i] = h[i]/l[i];
		z[i] = (A[i] - (h[i-1]*z[i-1]))/l[i];
	}

	// Step 5
	l[n] = 1.0;
	z[n] = 0.0;
	c[n] = 0.0;

	// Step 6
	for (int j = n - 1; j >= 0; j--)
	{
		c[j] = z[j]-u[j]*c[j+1];
		b[j] = (a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3.0;
		d[j] = (c[j+1]-c[j])/(3.0*h[j]);
	}
	// End cubic spline algorithm
	
	vector<double> roots;
	// Bisection method to find where cubic spline crosses baseline	
	for (int j = 0; j < data.size(); j++)
	{
		if ((fx(data[j].x, a, b, c, d, data)-baseline) * (fx(data[j+1].x, a, b, c, d, data)-baseline) < 0.0)
		{
			int i = 0;
			double FA = fx(data[j].x,a,b,c,d,data) - baseline;
			double x1 = data[j].x;
			double x2 = data[j+1].x;
			double p;
			double FP;
			int M = 50000;	// Maximum number of iterations
			do 
			{
				p = x1 + (x2-x1)/2.0;
				FP = fx(p, a, b, c, d, data) - baseline;
				if ((FP = 0) || ((x2-x1)/2.0 < tol))
				{
					roots.push_back(p);
					break;
				}

				i = i + 1;

				if (FA*FP > 0)
				{
					x1 = p;
					FA = FP;
				}

				else
					x2 = p;
			} while (i <= M); 
			
		} 
	}

	// Find location of all peaks and store them
	vector<double> peaks;
	for (int i = 0; i < roots.size(); i = i + 2)
		peaks.push_back((roots[i+1] + roots[i]) / 2);

	double area;		// Holds the value of the integral
	vector<double> areas;	// Vector that stores the integral of each peak
	double H;
	int N;

	if (method == 0)
	{
		N = 1000;
		int i; 
		vector<double> X1; X1.resize(1000);
		vector<double> TOL; TOL.resize(1000);
		vector<double> H1; H1.resize(1000);
		vector<double> FX1; FX1.resize(1000);
		vector<double> FX2; FX2.resize(1000);
		vector<double> FX3; FX3.resize(1000);
		vector<double> S; S.resize(1000);
		vector<int> L; L.resize(1000);
		double FD, FE, S1, S2;
		double u1, u2, u3, u4, u5, u6, u7;
		int u8;
		for (int j = 0; j < roots.size() - 1; j = j + 2)
		{
			i = 1;
			X1[i] = roots[j];
			H1[i] = (roots[j+1] - roots[j]) / 2.0;
			FX1[i] = fx(roots[j], a, b, c, d, data);
			FX2[i] = fx(roots[j] + H1[i], a, b, c, d, data);
			FX3[i] = fx(roots[j+1], a, b, c, d, data);
			area = 0.0;
			TOL[i] = 15.0*tol;
			S[i] = H1[i]*(FX1[i] + 4*FX2[i] + FX3[i])/3.0; // approximation for simpson's method of entire interval
			L[i] = 1;

			while ( i > 0 )
			{
				FD = fx(X1[i]+H1[i]/2.0, a, b, c, d, data);
				FE = fx(X1[i]+3.0*H1[i]/2.0, a, b, c, d, data);
				S1 = H1[i]*(FX1[i] + 4.0*FD + FX2[i])/6.0;
				S2 = H1[i]*(FX2[i] + 4.0*FE + FX3[i])/6.0; // S1 and S2 are simpson's method approximations for each half of the subintervals
				u1 = X1[i];
				u2 = FX1[i];
				u3 = FX2[i];
				u4 = FX3[i];
				u5 = H1[i];
				u6 = TOL[i];
				u7 = S[i];
				u8 = L[i];

				i = i - 1;
				if (fabs(u7 - S1 - S2) < u6)
				{
					area = area + (S1 + S2);
				}
				else
				{
					if (u8 >= N)
						break;
					else
					{
						i = i + 1; // data for left-half of subinterval
						X1[i] = u1 + u5;
						FX1[i] = u3;
						FX2[i] = FE;
						FX3[i] = u4;
						H1[i] = u5 / 2.0;
						TOL[i] = u6 / 2.0;
						S[i] = S2;
						L[i] = u8 + 1;

						i = i + 1; // data for right-half of subinterval
						X1[i] = u1;
						FX1[i] = u2;
						FX2[i] = FD;
						FX3[i] = u3;
						H1[i] = H1[i-1];
						TOL[i] = TOL[i-1];
						S[i] = S1;
						L[i] = L[i-1];
					}
				} 
			}
			areas.push_back(area);
		}		
	}

	// Trapezoidal Rule, h = x1 - x0
	else if (method == 1)
	{
		for (int i = 0; i < roots.size() - 1; i = i + 2)
		{
			H = roots[i+1] - roots[i];
			area = (H/2.0)*(fx(roots[i], a, b, c, d, data) + fx(roots[i+1], a, b, c, d, data));
			areas.push_back(area);		 	
		}
	}
	// Romberg Integration 
	else if (method == 2)
	{
		N = 4;
		double R[N][N];
		for (int i = 0; i < roots.size() - 1; i = i + 2)
		{
			H = roots[i+1] - roots[i];
			R[1][1] = (H/2.0)*(fx(roots[i], a, b, c, d, data) + fx(roots[i+1], a, b, c, d, data));
			for (int j = 2; j <= N; j++)
			{
				double trap = 0.0;
				for (int k = 1; k <= pow(2, j-2); k++)
					trap += (fx(roots[i]+(k-0.5)*H,a,b,c,d,data));

				R[2][1] = (1.0/2.0)*(R[1][1] + (H*trap));

				for (int M = 2; M <= j; M++)
					R[2][M] = R[2][M-1] + ((R[2][M-1]-R[1][M-1])/(pow(4.0,M-1)-1));

				H = H / 2.0;

				for (int M = 1; M <= j; M++)
					R[1][M] = R[2][M];
			}
			area = R[2][N-1];
			areas.push_back(area);		
		}
	}

	// Gaussian Quadrature
	else 
	{
		for (int i = 0; i < roots.size() - 1; i = i + 2)
		{
			area = fx(0.5*((roots[i+1]-roots[i])*0.5773502692+roots[i]+roots[i+1]), a, b, c, d, data) +
			       fx(0.5*((roots[i+1]-roots[i])*-0.5773502692+roots[i]+roots[i+1]), a, b, c, d, data); 

			areas.push_back(area);
		}
	}

	// Find number of hydrogens
	double minval = areas[0];
	double numhydros;
	vector<double> hydrogens;
	// Determine the smallest area
	for (int i = 0; i < areas.size(); i++)
	{
		if (areas[i] < minval)
			minval = areas[i];
	}

	for (int i = 0; i < areas.size(); i++)
	{
		numhydros = round(areas[i] / minval);
		hydrogens.push_back(numhydros);	
	}	

	// Analysis output
	cout << "	NMR ANALYSIS	" << endl;
	cout << "Program Options" << endl;
	cout << "========================" << endl;
	cout << "Baseline Adjustment	: " << baseline << endl;
	cout << "Tolerance		: " << tol << endl;
	cout << "Filtering		: " << filtertype << endl;
	cout << "Filter Size		: " << filtersize << endl;
	cout << "Filter Passes		: " << filterpasses << endl;
	cout << endl;
	cout << "Integration Method" << endl;
	cout << "========================" << endl;
	if (method == 0)
		cout << "Trapezoidal Rule" << endl;
	else if (method == 1)
		cout << "Romberg Integration" << endl;
	else if (method == 2)
		cout << "Adaptive Quadrature" << endl;
	else
		cout << "Gaussian Quadrature" << endl;

	cout << endl;
	cout << "Plot File Data" <<endl;
	cout << "========================" << endl;
	cout << "File: " << datafile << endl;
	cout << "Plot shifted " << tmspeak << " ppm for TMS peak calibration" << endl;

	cout << endl;
	cout << "Peak	Location	Area	Hydrogens" << endl;
	cout << "====== =============== ======= =========" << endl;
	for (int i = 0; i < peaks.size(); i++)
	{
	cout << i + 1 << "	" << peaks[i] << "	" << areas[i] << "	" << hydrogens[i] << endl;
	}
	
	return 0;
}
