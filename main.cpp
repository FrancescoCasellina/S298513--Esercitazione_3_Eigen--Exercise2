#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

VectorXd palu(const MatrixXd &A, const VectorXd &b) {
	PartialPivLU<MatrixXd> lu_decomp(A);
	return lu_decomp.solve(b);
}

VectorXd qr(const MatrixXd &A, const VectorXd &b) {
	HouseholderQR<MatrixXd> qr_decomp(A);
	return qr_decomp.solve(b);
}

double rel_err(const VectorXd &calc, const VectorXd X) {
	return ( calc - X ).norm() / X.norm();
}

int main () {
	MatrixXd A1(2,2), A2(2,2), A3(2,2);
	A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
	A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

	VectorXd b1(2), b2(2), b3(2);
	b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	b3 << -6.400391328043042e-10, 4.266924591433963e-10;

	vector<MatrixXd> As = {A1, A2, A3};
	vector<VectorXd> bs = {b1, b2, b3};

	VectorXd X(2);
	X << -1.0e+0, -1.0e+0;

	for(int n = 0; n < 3; n++) {
		cout << "Matrix A" << n+1 << ":\n" << As[n] << endl;
		cout << "Vector b" << n+1 << ":\n" << bs[n] << endl;

		cout << endl;

		cout << "Palu" << n+1 << ":\n" << palu(As[n], bs[n]) << endl;
		cout << "Relative Error:\n"  << rel_err(palu(As[n], bs[n]), X) << "\n" << endl;

		cout << "Qr"   << n+1 << ":\n" << qr(  As[n], bs[n]) << endl;
		cout << "Relative Error:\n"  << rel_err(qr  (As[n], bs[n]), X) << "\n" << endl;

		cout << "-------------o-------------\n" << endl;
	}


	return 0;
}
