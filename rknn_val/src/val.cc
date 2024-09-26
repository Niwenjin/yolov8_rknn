#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "file_utils.h"
#include "image_drawing.h"
#include "image_utils.h"
#include "yolov8.h"

const float THRESHOLD_IOU = 0.5;
// #define THRESHOLD_PROB 0.6
const int THRESHOLD_NUM = 7;
const float THRESHOLD_PROB[] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

// 定义类别结构体
typedef struct {
  int count = 0; // 类别的图片数量
  int correct[7] = {0};
  int error[7] = {0};
} class_stats_t;

typedef struct {
  int xmin;
  int ymin;
  int xmax;
  int ymax;
} box_t;

inline float CalculateOverlap(int xmin0, int ymin0, int xmax0, int ymax0,
                              int xmin1, int ymin1, int xmax1, int ymax1) {
  int w = std::max(0, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1);
  int h = std::max(0, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1);
  float intersection = (float)w * h; // 转换为浮点数以保持精度
  float iou = (float)(xmax0 - xmin0 + 1) * (ymax0 - ymin0 + 1) +
              (float)(xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1) - intersection;
  return iou <= 0.f ? 0.f : intersection / iou;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("%s <model_path> <data_path> <save_path>\n", argv[0]);
    return -1;
  }

  const char *model_path = argv[1];
  std::string data_path(argv[2]);
  std::string image_dir_str = data_path + "/images";
  std::string label_dir_str = data_path + "/labels";
  const char *image_dir = image_dir_str.c_str();
  const char *label_dir = label_dir_str.c_str();
  std::cout << "Image dir: " << image_dir << std::endl;
  std::cout << "Label dir: " << label_dir << std::endl;

  //   const char *image_dir = "/home/linaro/yolov8_val/build/image";
  //   const char *label_dir = "/home/linaro/yolov8_val/build/image";

  int ret;
  rknn_app_context_t rknn_app_ctx;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

  int count = 0;
  class_stats_t class_stats[9];

  init_post_process();

  ret = init_yolov8_model(model_path, &rknn_app_ctx);
  if (ret != 0) {
    printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
    goto out;
  }

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(image_dir)) != NULL) {
    // 遍历目录
    while ((ent = readdir(dir)) != NULL) {
      if (strstr(ent->d_name, ".jpg")) { // 检查文件扩展名是否为.jpg
        char image_path[256];
        sprintf(image_path, "%s/%s", image_dir, ent->d_name);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = read_image(image_path, &src_image);
        if (ret != 0) {
          printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
          goto out;
        }
        int height = src_image.height;
        int width = src_image.width;
        std::cout << "Image height: " << height << ", width: " << width
                  << std::endl;

        object_detect_result_list od_results;

        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
          printf("init_yolov8_model fail! ret=%d\n", ret);
          goto out;
        }
        if (od_results.count > 0)
          count++;
        for (int i = 0; i < od_results.count; i++) {
          object_detect_result *det_result = &(od_results.results[i]);
          printf("%d @ (%d %d %d %d) %.3f\n", det_result->cls_id,
                 det_result->box.left, det_result->box.top,
                 det_result->box.right, det_result->box.bottom,
                 det_result->prop);
        }

        // 读取标签文件
        char label_path[256];
        sprintf(label_path, "%s/%s", label_dir, ent->d_name);
        label_path[strlen(label_path) - 4] = '\0'; // 去掉.jpg，加上.txt
        strcat(label_path, ".txt");

        // FILE *label_file = fopen(label_path, "r");
        // if (label_file != NULL) {
        //   int label_class;
        //   float x, y, w, h;
        //   while (fscanf(label_file, "%d %f %f %f %f", &label_class, &x, &y,
        //   &w,
        //                 &h) != EOF) {
        //     class_stats[label_class].count++;
        //   }
        //   // 比较预测结果和标签
        //   for (int j = 0; j < od_results.count; j++) {
        //     object_detect_result *det_result = &(od_results.results[j]);
        //     if (det_result->prop > 0.5) {
        //       if (det_result->cls_id == label_class) {
        //         class_stats[label_class].correct++;
        //       } else {
        //         class_stats[label_class].error++;
        //       }
        //     }
        //   }
        //   fclose(label_file);
        // } else {
        //   printf("Label file not found: %s\n", label_path);
        // }

        FILE *label_file = fopen(label_path, "r");
        if (label_file != NULL) {
          int label_class;
          float x, y, w, h;
          std::vector<std::pair<int, box_t>>
              label_boxes; // 假设box_t是一个结构体，包含x, y, w, h

          while (fscanf(label_file, "%d %f %f %f %f", &label_class, &x, &y, &w,
                        &h) != EOF) {
            class_stats[label_class].count++;
            int xmin = int(x * width - w * width / 2);
            int ymin = int(y * height - h * height / 2);
            int xmax = int(x * width + w * width / 2);
            int ymax = int(y * height + h * height / 2);
            label_boxes.push_back(
                std::make_pair(label_class, box_t{xmin, ymin, xmax, ymax}));
          }

          // 比较预测结果和标签
          for (int i = 0; i < THRESHOLD_NUM; ++i) {
            for (int j = 0; j < od_results.count; j++) {
              object_detect_result *det_result = &(od_results.results[j]);
              if (det_result->prop > THRESHOLD_PROB[i]) {
                bool matched = false;
                for (auto &label_box : label_boxes) {
                  if (det_result->cls_id == label_box.first) {
                    float iou = CalculateOverlap(
                        det_result->box.left, det_result->box.top,
                        det_result->box.right, det_result->box.bottom,
                        label_box.second.xmin, label_box.second.ymin,
                        label_box.second.xmax, label_box.second.ymax);
                    //   printf("Box 1 - left: %d, bottom: %d, right: %d, top: "
                    //          "%d\n",
                    //          det_result->box.left, det_result->box.bottom,
                    //          det_result->box.right, det_result->box.top);
                    //   printf("Box 2 - xmin: %d, ymin: %d, xmax: %d, ymax: "
                    //          "%d\n",
                    //          label_box.second.xmin, label_box.second.ymin,
                    //          label_box.second.xmax, label_box.second.ymax);
                    //   printf("IOU result: %.4f\n", iou);
                    if (iou > THRESHOLD_IOU) {
                      class_stats[det_result->cls_id].correct[i]++;
                      matched = true;
                      break;
                    }
                  }
                }
                if (!matched) {
                  class_stats[det_result->cls_id].error[i]++;
                }
              }
            }
          }
          fclose(label_file);
        } else {
          printf("Label file not found: %s\n", label_path);
        }

        // 释放图片资源
        if (src_image.virt_addr != NULL) {
          // #if defined(RV1106_1103)
          //           dma_buf_free(rknn_app_ctx.img_dma_buf.size,
          //                        &rknn_app_ctx.img_dma_buf.dma_buf_fd,
          //                        rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
          // #else
          free(src_image.virt_addr);
          // #endif
        }
      }
    }
    closedir(dir);
  } else {
    printf("Could not open directory\n");
    return EXIT_FAILURE;
  }

out:
  deinit_post_process();

  ret = release_yolov8_model(&rknn_app_ctx);
  if (ret != 0) {
    printf("release_yolov8_model fail! ret=%d\n", ret);
  }

  printf("Total %d images detected\n", count);

  class_stats_t total_stats;
  // 打印每个类别的统计数据
  for (int j = 0; j < THRESHOLD_NUM; ++j) {
    printf("Threshold: %.2f\n", THRESHOLD_PROB[j]);
    for (int i = 0; i < 9; i++) {
      if (class_stats[i].count > 0) {
        if (j < 1)
          total_stats.count += class_stats[i].count;
        total_stats.correct[j] += class_stats[i].correct[j];
        total_stats.error[j] += class_stats[i].error[j];
        printf("Class %d: Count = %d, corr = %d, err = %d\n", i,
               class_stats[i].count, class_stats[i].correct[j],
               class_stats[i].error[j]);
      }
    }
    printf("total: Count = %d, corr = %d, err = %d\n", total_stats.count,
           total_stats.correct[j], total_stats.error[j]);
  }
  // 写入统计数据
  std::string filename(argv[3]);
  std::ofstream file_out(filename);

  // 检查文件是否成功打开
  if (!file_out.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return -1;
  }

  // 写入数据到文件
  for (int j = 0; j < THRESHOLD_NUM; ++j) {
    file_out << "Threshold: " << std::fixed << std::setprecision(2)
             << THRESHOLD_PROB[j] << std::endl;
    for (int i = 0; i < 9; i++) {
      if (class_stats[i].count > 0) {
        // if (j < 1)
        //   total_stats.count += class_stats[i].count;
        // total_stats.correct[j] += class_stats[i].correct[j];
        // total_stats.error[j] += class_stats[i].error[j];
        file_out << "Class " << i << ": Count = " << class_stats[i].count
                 << ", corr = " << class_stats[i].correct[j]
                 << ", err = " << class_stats[i].error[j] << std::endl;
      }
    }
    file_out << "total: Count = " << total_stats.count
             << ", corr = " << total_stats.correct[j]
             << ", err = " << total_stats.error[j] << "\n"
             << std::endl;
  }

  // 关闭文件
  file_out.close();
  std::cout << "Data has been written to file: " << filename << std::endl;

  return 0;
}