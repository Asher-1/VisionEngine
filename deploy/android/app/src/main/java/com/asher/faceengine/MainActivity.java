// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.asher.faceengine;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;

    private int facing = 0;
    private static final int SELECT_IMAGE = 1;
    private SurfaceView cameraView;
    private visionEngineNcnn engineNcnn = new visionEngineNcnn();

    private int currentDetModel = 0;
    private int currentRecModel = 0;
    private int recEnabled = 0;
    private int liveEnabled = 0;
    private int kpsEnabled = 0;
    private int gpuEnabled = 0;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                engineNcnn.closeCamera();
                facing = 1 - facing;
                openCamera();
            }
        });

        Button buttonRegisterFace = (Button) findViewById(R.id.buttonRegisterFace);
        buttonRegisterFace.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (recEnabled == 0) {
                    Toast.makeText(getApplicationContext(), "请先开启人脸识别功能",
                            Toast.LENGTH_SHORT).show();
                    return;
                }

                engineNcnn.closeCamera();

                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        Button buttonDeleteFace = (Button) findViewById(R.id.buttonDeleteFace);
        buttonDeleteFace.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                CustomDialog.showDialog(MainActivity.this,
                        "请输入待删除人员名称", new CustomDialog.ResultCallBack() {
                            @Override
                            public void callback(String name) {
                                if (name == null || name.isEmpty()) {
                                    Toast.makeText(getApplicationContext(), "输入信息有误或者取消删除",
                                            Toast.LENGTH_SHORT).show();
                                    return;
                                }

                                if (engineNcnn.deleteFace(name)) {
                                    Toast.makeText(getApplicationContext(), "删除人员注册信息成功",
                                            Toast.LENGTH_SHORT).show();
                                } else {
                                    Toast.makeText(getApplicationContext(), "删除人员注册信息失败",
                                            Toast.LENGTH_SHORT).show();
                                }
                            }
                        });
            }
        });

        Button buttonResetFaces = (Button) findViewById(R.id.buttonResetFaces);
        buttonResetFaces.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                new AlertDialog.Builder(MainActivity.this).setTitle("是否确定重置人员注册信息?")
                        .setIcon(android.R.drawable.ic_dialog_alert)
                        .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface arg0, int arg1) {
                                if (engineNcnn.clearFaces()) {
                                    Toast.makeText(getApplicationContext(), "重置人员注册信息成功",
                                            Toast.LENGTH_SHORT).show();
                                } else {
                                    Toast.makeText(getApplicationContext(), "删除人员注册信息失败",
                                            Toast.LENGTH_SHORT).show();
                                }
                            }
                        }).setNegativeButton("取消", null).show();
            }
        });

        Spinner detModel = (Spinner) findViewById(R.id.detModel);
        detModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != currentDetModel) {
                    currentDetModel = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Spinner recModel = (Spinner) findViewById(R.id.recModel);
        recModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != currentRecModel) {
                    currentRecModel = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Spinner toggleCpuGpuModel = (Spinner) findViewById(R.id.toggleCpuGpuModel);
        toggleCpuGpuModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != gpuEnabled) {
                    gpuEnabled = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Spinner toggleRecModel = (Spinner) findViewById(R.id.toggleRec);
        toggleRecModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != recEnabled) {
                    recEnabled = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Spinner toggleLiveModel = (Spinner) findViewById(R.id.toggleLive);
        toggleLiveModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != liveEnabled) {
                    liveEnabled = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Spinner toggleKpsModel = (Spinner) findViewById(R.id.toggleKps);
        toggleKpsModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != kpsEnabled) {
                    kpsEnabled = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        reload();
    }

    private void reload() {
        String dbPath = getFilesDir().getPath();
        File filePath = new File(dbPath);
        if (!filePath.exists()) {
            if (!filePath.mkdirs()) {
                Log.e("MainActivity", "engineNcnn create db path failed!");
            }
        }
        boolean ret_init = engineNcnn.loadModel(getAssets(), dbPath,
                currentDetModel, currentRecModel, recEnabled, liveEnabled, kpsEnabled, gpuEnabled);
        if (!ret_init) {
            Toast.makeText(getApplicationContext(), "重新加载模型失败!",
                    Toast.LENGTH_SHORT).show();
            Log.e("MainActivity", "engineNcnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        engineNcnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        engineNcnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        engineNcnn.closeCamera();
    }

    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        openCamera();
    }

    public boolean openCamera() {
        return engineNcnn.openCamera(facing);
    }

    @Override
    public void onPause() {
        super.onPause();
        engineNcnn.closeCamera();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);
                    final Bitmap registeredImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    if (registeredImage == null) {
                        Toast.makeText(getApplicationContext(), "注册图像非法!",
                                Toast.LENGTH_SHORT).show();
                        openCamera();
                        return;
                    }

                    CustomDialog.showDialog(MainActivity.this,
                            "请输入注册人员名称", new CustomDialog.ResultCallBack() {
                                @Override
                                public void callback(String name) {
                                    if (name == null || name.isEmpty()) {
                                        Toast.makeText(MainActivity.this, "输入信息有误或者取消注册",
                                                Toast.LENGTH_SHORT).show();
                                        return;
                                    }

                                    // test code
//                                    KeyPoint[] keyPoints = engineNcnn.detectFaces(registeredImage, 0.6f, 0.45f);
//                                    for (KeyPoint keyPoint : keyPoints) {
//                                        boolean isSame = engineNcnn.verifyFace(registeredImage, keyPoint);
//                                        String text = isSame ? "验证通过" : "验证失败";
//                                        Toast.makeText(getApplicationContext(), text,
//                                                Toast.LENGTH_SHORT).show();
//                                        LivingInfo livingInfo = engineNcnn.detectLiving(registeredImage, keyPoint);
//                                        text = livingInfo.isLiving ? String.format("real face %.2f", livingInfo.score) :
//                                                String.format("fake face %.2f", livingInfo.score);
//                                        Toast.makeText(getApplicationContext(), text,
//                                                Toast.LENGTH_SHORT).show();
//                                    }

                                    if (engineNcnn.registerFace(registeredImage, name)) {
                                        Toast.makeText(MainActivity.this, "注册成功",
                                                Toast.LENGTH_SHORT).show();
                                    } else {
                                        Toast.makeText(MainActivity.this, "注册失败",
                                                Toast.LENGTH_SHORT).show();
                                    }

                                    List<String> names = new ArrayList<>();
                                    if (engineNcnn.findFaces(names)) {
                                        @SuppressLint("DefaultLocale") String text =
                                                String.format("已注册 %d 名人员", names.size());
                                        Toast.makeText(getApplicationContext(), text,
                                                Toast.LENGTH_SHORT).show();
                                    } else {
                                        Toast.makeText(getApplicationContext(), "未找到注册人员",
                                                Toast.LENGTH_SHORT).show();
                                    }
                                }
                            });
                }
            } catch (FileNotFoundException e) {
                Log.e("MainActivity", "FileNotFoundException");
            } finally {
                openCamera();
            }
        } else {
            openCamera();
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 640;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (width_tmp / 2 >= REQUIRED_SIZE
                && height_tmp / 2 >= REQUIRED_SIZE) {
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        } catch (IOException e) {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        assert bitmap != null;
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(),
                bitmap.getHeight(), matrix, true);
    }

}
