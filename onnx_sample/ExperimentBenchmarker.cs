using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using System.Drawing;
using Emgu.CV.Reg;

namespace onnx_sample
{
    [MemoryDiagnoser]
    public class ExperimentBenchhmarker
    {
        static string[] imagesPaths = { "./images/000001.jpg", "./images/000002.jpg", "./images/000003.jpg" };
        static Bitmap image = new Bitmap(imagesPaths[0]);
        private static List<NamedOnnxValue> expFunction(Bitmap img)
        {
            int[] inputDimension = [1, 3, 640, 640];
            img = new Bitmap(img, new Size(640, 640));
            var ndarray = img.ToNDArray(flat: false, copy: false);
            ndarray /= 255;
            ndarray.transpose(new int[] { 2, 0, 1 });
            Tensor<float> input = new DenseTensor<float>(memory: ndarray.ToArray<float>(), inputDimension);
            return new List<NamedOnnxValue>();
        }

        [Benchmark]
        public void RunExpFunction()
        {
            var res = expFunction(image);

        }


    }

}
