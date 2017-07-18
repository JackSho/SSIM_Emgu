using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

//using ImageDiff;

namespace EmguTest
{
	class Program
	{
		static void Main(string[] args)
		{
			if (args.Length != 2)
			{
				Console.WriteLine("Usage: EmguTest <img1> <img2>");
				return;
			}

			string str1 = args[0];
			string str2 = args[1];

			try
			{
				ImageDiff.ImageSSIM imagediff = new ImageDiff.ImageSSIM(str1, str2);
				double ssim = imagediff.CalcSSIM();//相似度[0，1]， 等于1代表图片完全相同
				double ssimRed = imagediff.SSIMRed;
				double ssimGreen = imagediff.SSIMGreen;
				double ssimBlue = imagediff.SSIMBlue;
				int num = imagediff.NumOfDifferences;//不同的区域个数
				Console.WriteLine("SSIM_All: {0}", ssim);
				Console.WriteLine("SSIM_Red: {0}", ssimRed);
				Console.WriteLine("SSIM_Green: {0}", ssimGreen);
				Console.WriteLine("SSIM_Blue: {0}", ssimBlue);
			}
			catch (ImageDiff.ImageDiffException ex)
			{
				string msg = ex.Message;
			}

		}
	}
}
