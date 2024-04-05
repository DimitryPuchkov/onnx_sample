using MathNet.Filtering.Kalman;
using MathNet.Numerics.LinearAlgebra;


namespace onnx_sample
{
    public class KalmanBoxTracker
    {
        Matrix<double> F;
        Matrix<double> Q;
        Matrix<double> H;
        Matrix<double> R;
        DiscreteKalmanFilter kf;
        const double stdWeightPosition = 1.0 / 20.0;
        const double stdWeightVelocity = 1.0 / 160.0;
        public int TimeScinceUpdate { get; set; }
        private static int count = 0;
        public int TrackerID { get; set; }
        public KalmanBoxTracker(Box box) 
        {
            KalmanBoxTracker.count++;
            TrackerID = count;
            var M = Matrix<double>.Build;
            Matrix<double> x0 = M.Dense(8, 1, 0);
            var z = convertBoxToZ(box);
            x0[0, 0] = z[0, 0];
            x0[1, 0] = z[1, 0];
            x0[2, 0] = z[2, 0];
            x0[3, 0] = z[3, 0];
            Matrix<double> P0 = M.Diagonal([1, 1, 1, 1, 1000, 1000, 1000, 10000]);

            //F
            double[,] fArr = {
                { 1, 0, 0, 0, 1, 0, 0, 0},
                { 0, 1, 0, 0, 0, 1, 0, 0},
                { 0, 0, 1, 0, 0, 0, 1, 0},
                { 0, 0, 0, 1, 0, 0, 0, 1},
                { 0, 0, 0, 0, 1, 0, 0, 0},
                { 0, 0, 0, 0, 0, 1, 0, 0},
                { 0, 0, 0, 0, 0, 0, 1, 0},
                { 0, 0, 0, 0, 0, 0, 0, 1},
            };
            F = M.DenseOfArray(fArr);

            // H
            double[,] HArr = {
                { 1, 0, 0, 0, 0, 0, 0, 0},
                { 0, 1, 0, 0, 0, 0, 0, 0},
                { 0, 0, 1, 0, 0, 0, 0, 0},
                { 0, 0, 0, 1, 0, 0, 0, 0},

            };
            H = M.DenseOfArray(HArr);

            Q = M.Dense(8, 8);
            R = M.Dense(4, 4);
            kf = new DiscreteKalmanFilter(x0, P0);

            TimeScinceUpdate = 0;
        }

        public Box GetState()
        {
            var x = kf.State;
            Box box = new Box();
            double w = x[2, 0];
            double h = x[3, 0];
            box.x1 = (int)(x[0, 0] - w / 2);
            box.x2 = (int)(x[0, 0] + w / 2);
            box.y1 = (int)(x[1, 0] - h / 2);
            box.y2 = (int)(x[1, 0] + h / 2);
            return box;
        }

        public void Predict()
        {
            var x = kf.State;
            if (x[2, 0] + x[6, 0] < 0)
                x[6, 0] *= 0.0;

            Q[0, 0] = Math.Pow(stdWeightPosition * x[2, 0], 2);
            Q[1, 1] = Math.Pow(stdWeightPosition * x[3, 0], 2);
            Q[2, 2] = Math.Pow(stdWeightPosition * x[2, 0], 2);
            Q[3, 3] = Math.Pow(stdWeightPosition * x[3, 0], 2);
            Q[4, 4] = Math.Pow(stdWeightVelocity * x[2, 0], 2);
            Q[5, 5] = Math.Pow(stdWeightVelocity * x[3, 0], 2);
            Q[6, 6] = Math.Pow(stdWeightVelocity * x[2, 0], 2);
            Q[7, 7] = Math.Pow(stdWeightVelocity * x[3, 0], 2);
            
            kf.Predict(F, Q);
            TimeScinceUpdate += 1;
        }

        public void Update(Box box)
        {
            var x = kf.State;
      
            R[0, 0] = Math.Pow(stdWeightPosition * x[2, 0], 2);
            R[1, 1] = Math.Pow(stdWeightPosition * x[3, 0], 2);
            R[2, 2] = Math.Pow(stdWeightPosition * x[2, 0], 2);
            R[3, 3] = Math.Pow(stdWeightPosition * x[3, 0], 2);

            TimeScinceUpdate = 0;

            kf.Update(convertBoxToZ(box), H, R);
        }

        private Matrix<double> convertBoxToZ(Box box)
        {
            Matrix<double> z = Matrix<double>.Build.Dense(4, 1);
            z[2, 0] = box.x2 - box.x1;
            z[3, 0] = box.y2 - box.y1;
            z[0, 0] = box.x1 + z[2, 0]/2;
            z[1, 0] = box.y1 + z[3, 0]/2;
            return z;
        }




    }
}
