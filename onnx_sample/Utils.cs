using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnx_sample
{
    public static class Utils
    {
        public static List<Box> NMS(ref List<Box> correct, float thr = 0.5f)
        {
            if (correct.Count == 0)
                return new List<Box>();
            correct = correct.OrderBy(x => x.score).ToList();
            var final_boxes = new List<Box>();
            while (correct.Count > 0)
            {
                var current_box = correct[correct.Count - 1];
                correct.RemoveAt(correct.Count - 1);
                final_boxes.Add(current_box);

                correct = correct.Where(box => IOU(current_box, box) <= thr).ToList();
            }
            return final_boxes;
        }

        public static float IOU(Box box1, Box box2)
        {
            float x_overlap = Math.Max(0, Math.Min(box1.x2, box2.x2) - Math.Max(box1.x1, box2.x1));
            float y_overlap = Math.Max(0, Math.Min(box1.y2, box2.y2) - Math.Max(box1.y1, box2.y1));
            float overlap_area = x_overlap * y_overlap;
            float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
            float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
            float total_area = area1 + area2 - overlap_area;
            return overlap_area / total_area;
        }
    }
}
