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
using System.Drawing.Imaging;

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
            img = img.Clone(new Rectangle(0, 0, img.Width, img.Height), PixelFormat.Format24bppRgb);
            var ndarray = img.ToNDArray(flat: false, copy: false);
            ndarray = ndarray.astype(typeof(float));
            ndarray /= 255;
            ndarray = ndarray.transpose(new int[] { 0, 3, 1, 2 });
            Tensor<float> inputTensor = new DenseTensor<float>(memory: ndarray.ToArray<float>(), inputDimension);
            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
        }

        [Benchmark]
        public void RunExpFunction()
        {
            var res = expFunction(image);

        }


    }

}
