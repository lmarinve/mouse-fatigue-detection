       /////////////////////////////////////////////
      // Mouse Fatigue Estimation by GSR and EMG //
     //          Values w/ TensorFlow           //
    //             ---------------             //
   //              (Wio Terminal)             //
  //             by Kutluhan Aktar           //
 //                                         //
/////////////////////////////////////////////

//
// Collate forearm muscle soreness data on the SD card, build and train a neural network model, and run the model directly on Wio Terminal.
//
// For more information:
// https://www.theamplituhedron.com/projects/Mouse_Fatigue_Estimation_by_GSR_and_EMG_Values_w_TensorFlow/
//
//
// Connections
// Wio Terminal :
//                                Grove - GSR sensor
// A0  --------------------------- Grove Connector
//                                Grove - EMG Detector
// A2  --------------------------- Grove Connector


// Include the required libraries.
#include <SPI.h>
#include <Seeed_FS.h>
#include "TFT_eSPI.h"
#include "seeed_line_chart.h"
#include "SD/Seeed_SD.h"
#include "RawImage.h"

// Import the required TensorFlow modules.
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/version.h"

// Import the converted TensorFlow Lite model.
#include "mouse_fatigue_level.h"

// TFLite globals, used for compatibility with Arduino-style sketches:
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow arrays.
  constexpr int kTensorArenaSize = 15 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// Define the threshold value for the model outputs (results).
float threshold = 0.75;

// Define the muscle soreness level (class) names and color codes:
String classes[] = {"Relaxed", "Tense", "Exhausted"};
uint32_t color_codes[] = {tft.color565(1,156,0), tft.color565(255,169,2), tft.color565(226,16,1)};

// Define the class image list:
const char* images[] = {"relaxed.bmp", "tense.bmp", "exhausted.bmp"};

// Define the TFT screen:
TFT_eSPI tft;

// Define the sprite settings: 
#define max_size 50 // maximum size of data
doubles gsr_data, emg_data;
TFT_eSprite spr = TFT_eSprite(&tft);

// Define the sensor voltage (signal) pins:
#define GSR A0
#define EMG A2

// Define the data holders.
int gsr_value, emg_value;
uint32_t background_color = tft.color565(31,32,32);
uint32_t text_color = tft.color565(174,255,205);

void setup(){
  Serial.begin(115200);

  // 5-Way Switch
  pinMode(WIO_5S_PRESS, INPUT_PULLUP);

  // TensorFlow Lite Model settings:
  
  // Set up logging (will report to Serial, even within TFLite functions).
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure.
  model = tflite::GetModel(mouse_fatigue_level);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  delay(1000);

  // Check the connection status between Wio Terminal and the SD card.
  if(!SD.begin(SDCARD_SS_PIN, SDCARD_SPI)) while (1);

  // Initiate the TFT screen:
  tft.begin();
  tft.setRotation(3);
  tft.fillScreen(background_color);
  tft.setTextColor(text_color);
  tft.setTextSize(2);
  // Create the sprite.
  spr.createSprite(TFT_HEIGHT / 2, TFT_WIDTH);

  // Define and display the 16-bit images saved on the SD card:
  for(int i=0; i<3; i++){ drawImage<uint16_t>(images[i], TFT_HEIGHT, 0); }
  drawImage<uint16_t>("carpal_tunnel.bmp", TFT_HEIGHT/2, 0);
  drawImage<uint16_t>("mouse.bmp", TFT_HEIGHT/2, TFT_WIDTH-90);

  delay(1000);
}

void loop(){
  // Obtain current measurements generated by the GSR sensor and the EMG sensor.
  get_GSR_data(3);
  get_EMG_data();
  // Initialize the sprite.
  spr.fillSprite(background_color);
  
  // Adjust the line chart data arrays:
  if(gsr_data.size() == max_size) gsr_data.pop();
  if(emg_data.size() == max_size) emg_data.pop();
  
  // Append new data variables to the line chart data arrays:
  gsr_data.push(gsr_value);
  emg_data.push(emg_value);
  
  // Display the line charts on the TFT screen: 
  display_line_chart(0, "GSR", TFT_HEIGHT/2, 90, gsr_data, text_color, tft.color565(165,40,44));
  display_line_chart(110, "EMG", TFT_HEIGHT/2, 90, emg_data, text_color, tft.color565(165,40,44));
  spr.pushSprite(0, 0);
  spr.setTextColor(text_color);
  delay(50);

  // Execute the TensorFlow Lite model to make predictions on the muscle soreness levels (classes). 
  if(digitalRead(WIO_5S_PRESS) == LOW) run_inference_to_make_predictions();
}

void run_inference_to_make_predictions(){    
    // Scale (normalize) muscle soreness data depending on the given model and copy them to the input buffer (tensor):
    model_input->data.f[0] = gsr_value / 1000;
    model_input->data.f[1] = emg_value / 1000;
    
    // Run inference:
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on the given input.");
    }

    // Read predicted y values (muscle soreness classes) from the output buffer (tensor): 
    for(int i = 0; i<3; i++){
      if(model_output->data.f[i] >= threshold){
        int w = 150, h = 40, str_x = 12, str_y = 65;
        int y_offset = h + ((h - str_y) / 2);
        int x_offset = classes[i].length() * str_x;
        // Display the detection result (class).
        tft.fillScreen(background_color);
        drawImage<uint16_t>(images[i], (TFT_HEIGHT-75)/2, 0);
        // Print:
        tft.fillRect((TFT_HEIGHT-w)/2, TFT_WIDTH-h, w, h, color_codes[i]);
        tft.drawString(classes[i], (TFT_HEIGHT-x_offset)/2, TFT_WIDTH-y_offset);
      }
    }
    // Exit and clear:
    delay(3000);
    tft.fillScreen(background_color);
    drawImage<uint16_t>("carpal_tunnel.bmp", TFT_HEIGHT/2, 0);
    drawImage<uint16_t>("mouse.bmp", TFT_HEIGHT/2, TFT_WIDTH-90);
}

void display_line_chart(int header_y, const char* header_title, int chart_width, int chart_height, doubles data, uint32_t graph_color, uint32_t line_color){
  // Define the line graph title settings:
  auto header =  text(0, header_y)
                 .value(header_title)
                 .align(center)
                 .valign(vcenter)
                 .width(chart_width)
                 .color(tft.color565(243,208,296))
                 .thickness(2);
  // Define the header height and draw the line graph title. 
  header.height(header.font_height() * 2);
  header.draw();
  // Define the line chart settings:
  auto content = line_chart(0, header.height() + header_y);
  content
  .height(chart_height) // the actual height of the line chart
  .width(chart_width) // the actual width of the line chart
  .based_on(0.0) // the starting point of the y-axis must be float
  .show_circle(false) // drawing a circle at each point, default is on
  .value(data) // passing the given data array to the line graph
  .color(line_color) // setting the line color 
  .x_role_color(graph_color) // setting the line graph color
  .x_tick_color(graph_color)
  .y_role_color(graph_color)
  .y_tick_color(graph_color)
  .draw();
}

void get_GSR_data(int calibration){
  long sum = 0;
  // Calculate the average of the last ten GSR sensor measurements to remove the glitch.
  for(int i=0;i<10;i++){
    sum += analogRead(GSR);
    delay(5);
  }
  gsr_value = (sum / 10) - calibration;
  Serial.print("GSR Value => "); Serial.println(gsr_value);
}

void get_EMG_data(){
  long sum = 0;
  // Evaluate the summation of the last 32 EMG sensor measurements.
  for(int i=0;i<32;i++){
    sum += analogRead(EMG);
  }
  // Shift the summation by five with the right shift operator (>>) to obtain the EMG value.  
  emg_value = sum >> 5;
  Serial.print("EMG Value => "); Serial.println(emg_value); Serial.println();
  delay(10);
}
