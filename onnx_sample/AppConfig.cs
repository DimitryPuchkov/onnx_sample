using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnx_sample
{
    public static class AppConfig
    {
        public static string ModelPath { get; } = "./mot17-01-frcnn.onnx";
        public static int ModelInputSize { get; } = 640;
    }
}
