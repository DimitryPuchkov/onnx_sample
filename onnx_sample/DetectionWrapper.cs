using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime.Tensors;
using static System.Net.Mime.MediaTypeNames;
using SixLabors.ImageSharp.Processing;

namespace onnx_sample
{

    public struct Box
    {
        public int x1, x2, y1, y2, object_id;
        public float score;

        public Box()
        {
            object_id = -1;
        }
    }

    public class DetectionWrapper
    {
        private InferenceSession session = new InferenceSession(AppConfig.ModelPath);

        public List<Box> Detect(Image<Rgb24> image)
        {
            image = image.Clone();
            double scale_x, scale_y;
            scale_x = (double)image.Width / (double)AppConfig.ModelInputSize;
            scale_y = (double)image.Height / (double)AppConfig.ModelInputSize;

            var modelInput = Preprocess(image);
            var modelOutput = session.Run(modelInput);

            return Postprocess(modelOutput[0].AsTensor<float>(), scale_x, scale_y);

        }

        private List<NamedOnnxValue> Preprocess(Image<Rgb24> image)
        {
            image.Mutate(x => x.Resize(AppConfig.ModelInputSize, AppConfig.ModelInputSize));
            int[] inputDimension = [1, 3, 640, 640];

            Tensor<float> input = new DenseTensor<float>(inputDimension);

            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (var x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = (float)pixelSpan[x].R / 255.0f;
                        input[0, 1, y, x] = (float)pixelSpan[x].G / 255.0f;
                        input[0, 2, y, x] = (float)pixelSpan[x].B / 255.0f;
                    }
                }
            });

            return new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor("images", input)};
        }

        private List<Box> Postprocess(Tensor<float> outTensor, double scale_x, double scale_y)
        {
            var correct = new List<Box>();
            for (int i = 0; i < outTensor.Dimensions[2]; i++)
            {
                if (outTensor[0, 4, i] > 0.3)
                {
                    var box = new Box();
                    box.x1 = (int)((outTensor[0, 0, i] - outTensor[0, 2, i] / 2) * scale_x);
                    box.x2 = (int)((outTensor[0, 0, i] + outTensor[0, 2, i] / 2) * scale_x);
                    box.y1 = (int)((outTensor[0, 1, i] - outTensor[0, 3, i] / 2) * scale_y);
                    box.y2 = (int)((outTensor[0, 1, i] + outTensor[0, 3, i] / 2) * scale_y);
                    box.score = outTensor[0, 4, i];
                    correct.Add(box);
                }
            }
            return Utils.NMS(ref correct);
        }
    }
}
