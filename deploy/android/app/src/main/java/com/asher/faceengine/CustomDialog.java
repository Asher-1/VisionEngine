package com.asher.faceengine;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.view.KeyEvent;
import android.widget.EditText;

public class CustomDialog {
    private static AlertDialog dialog;

    //回调接口
    public interface ResultCallBack {
        void callback(String name);
    }

    public static void showDialog(Context context, String str, final ResultCallBack mRCallBack) {
        final EditText et = new EditText(context);
        dialog = new AlertDialog.Builder(context)
                .setTitle("提示")
                .setIcon(android.R.drawable.ic_dialog_info).setView(et)
                .setMessage(str)
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (mRCallBack != null) {
                            String name = et.getText().toString();
                            mRCallBack.callback(name);
                        }
                    }
                })
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (mRCallBack != null) {
                            mRCallBack.callback("");
                        }
                    }
                }).create();
        dialog.setCancelable(false);
        dialog.setOnKeyListener(new DialogInterface.OnKeyListener() {
            @Override
            public boolean onKey(DialogInterface dialog, int keyCode, KeyEvent event) {
                if (keyCode == KeyEvent.KEYCODE_SEARCH) {//屏蔽搜索键
                    return true;
                } else {
                    return false; //默认返回 false
                }
            }
        });
        dialog.show();
    }
}