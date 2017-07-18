using System;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace ImageDiff
{
	public class ImageDiffException : ApplicationException
	{
		public ImageDiffException() { }

		public ImageDiffException(string message) : base(message) { }
	}

	public class ImageSSIM
	{
		private enum RGBIndex
		{
			Red = 0,
			Green,
			Blue,
			All
		}
		private double[] SSIM = new double[] { -1, -1, -1, -1 };
		private static ImageDiffException notCalcedException = new ImageDiffException("The SSIM have not been calculated yet.");
		private const double C1 = 6.5025;
		private const double C2 = 58.5225;
		private int NumDifferences = 0;

		/// <summary>
		/// 获取 Red 通道的 SSIM， 如果未计算 SSIM ，会抛异常 ImageDiffException
		/// </summary>
		public double SSIMRed 
		{
			get
			{
				if (SSIM[(int)RGBIndex.Red] == -1)
					throw notCalcedException;
				return SSIM[(int)RGBIndex.Red];
			}
		}

		/// <summary>
		/// 获取 Green 通道的 SSIM， 如果未计算 SSIM ，会抛异常 ImageDiffException
		/// </summary>
		public double SSIMGreen
		{
			get
			{
				if (SSIM[(int)RGBIndex.Green] == -1)
					throw notCalcedException;
				return SSIM[(int)RGBIndex.Green];
			}
		}
		
		/// <summary>
		/// 获取 Blue 通道的 SSIM， 如果未计算 SSIM ，会抛异常 ImageDiffException
		/// </summary>
		public double SSIMBlue
		{
			get
			{
				if (SSIM[(int)RGBIndex.Blue] == -1)
					throw notCalcedException;
				return SSIM[(int)RGBIndex.Blue];
			}
		}

		/// <summary>
		/// 获取所有通道的 SSIM， 如果未计算 SSIM ，会抛异常 ImageDiffException
		/// </summary>
		public double SSIMAll
		{
			get
			{
				if (SSIM[(int)RGBIndex.All] == -1)
					throw notCalcedException;
				return SSIM[(int)RGBIndex.All];
			}
		}

		/// <summary>
		/// 获取不同区域的数量，如果未计算 SSIM ，会抛异常 ImageDiffException
		/// </summary>
		public int NumOfDifferences
		{
			get
			{
				if (SSIM[(int)RGBIndex.All] == -1)
					throw notCalcedException;
				return NumDifferences;
			}
		}

		/// <summary>
		/// 获取或设置矩形标记的颜色，默认为 Color.Red
		/// </summary>
		public Color RectColor { get; set; }

		/// <summary>
		/// 获取或设置第一张图片的路径
		/// </summary>
		public string Image1 { get; set; }

		/// <summary>
		/// 获取或设置第二张图片的路径
		/// </summary>
		public string Image2 { get; set; }

		/// <summary>
		/// 获取或设置要生成的有不同标记的图片的路径
		/// </summary>
		public string ImageDifferent { get; set; }

		/// <summary>
		/// 构造一个 ImageSSIM 对象，不指定任何路径
		/// </summary>
		public ImageSSIM()
		{
			Image1 = null;
			Image2 = null;
			ImageDifferent = null;
			NumDifferences = 0;
			RectColor = Color.Red;
		}

		/// <summary>
		/// 构造一个 ImageSSIM 对象
		/// </summary>
		/// <param name="image1">第一张图片的路径</param>
		/// <param name="image2">第二张图片的路径</param>
		public ImageSSIM(string image1, string image2):this()
		{
			Image1 = image1;
			Image2 = image2;
		}
		
		/// <summary>
		/// 构造一个 ImageSSIM 对象
		/// </summary>
		/// <param name="image1">第一张图片的路径</param>
		/// <param name="image2">第二张图片的路径</param>
		/// <param name="imageDiffenent">如果图片不同，需要生成的图片路径</param>
		public ImageSSIM(string image1, string image2, string imageDiffenent):this(image1, image2)
		{
			ImageDifferent = imageDiffenent;
		}

		/// <summary>
		/// 计算两张图片的相似度，取值范围[0，1]， 等于1代表完全相同
		/// </summary>
		/// <returns>返回两张图片的相似度</returns>
		public double CalcSSIM()
		{
			if (SSIM[(int)RGBIndex.All] != -1)
				return SSIM[(int)RGBIndex.All];

			if (Image1 == null)
				throw new ImageDiffException("image1 can not be null.");
			if (Image2 == null)
				throw new ImageDiffException("image2 can not be null.");

			Image<Bgr, Byte> img1_temp = null;
			Image<Bgr, Byte> img2_temp = null;
			try
			{
				img1_temp = new Image<Bgr, Byte>(Image1);
				img2_temp = new Image<Bgr, Byte>(Image2);
			}
			catch (Exception ex)
			{
				throw new ImageDiffException(ex.Message);
			}

			int imageWidth = img1_temp.Width;
			int imageHeight = img1_temp.Height;
			int nChan = img1_temp.NumberOfChannels;
			IplDepth depth32F = IplDepth.IplDepth32F;
			Size imageSize = new Size(imageWidth, imageHeight);

			Image<Bgr, Single> img1 = img1_temp.ConvertScale<Single>(1.0, 1);
			Image<Bgr, Single> img2 = img2_temp.ConvertScale<Single>(1.0, 1);
			Image<Bgr, Byte> diff = img2_temp.Copy();

			Image<Bgr, Single> img1_sq = img1.Pow(2);
			Image<Bgr, Single> img2_sq = img2.Pow(2);
			Image<Bgr, Single> img1_img2 = img1.Mul(img2);

			Image<Bgr, Single> mu1 = img1.SmoothGaussian(11, 11, 1.5, 0);
			Image<Bgr, Single> mu2 = img2.SmoothGaussian(11, 11, 1.5, 0);

			Image<Bgr, Single> mu1_sq = mu1.Pow(2);
			Image<Bgr, Single> mu2_sq = mu2.Pow(2);
			Image<Bgr, Single> mu1_mu2 = mu1.Mul(mu2);

			Image<Bgr, Single> sigma1_sq = img1_sq.SmoothGaussian(11, 11, 1.5, 0);
			sigma1_sq = sigma1_sq.AddWeighted(mu1_sq, 1, -1, 0);

			Image<Bgr, Single> sigma2_sq = img2_sq.SmoothGaussian(11, 11, 1.5, 0);
			sigma2_sq = sigma2_sq.AddWeighted(mu2_sq, 1, -1, 0);

			Image<Bgr, Single> sigma12 = img1_img2.SmoothGaussian(11, 11, 1.5, 0);
			sigma12 = sigma12.AddWeighted(mu1_mu2, 1, -1, 0);

			// (2*mu1_mu2 + C1)
			Image<Bgr, Single> temp1 = mu1_mu2.ConvertScale<Single>(2, 0);
			temp1 = temp1.Add(new Bgr(C1, C1, C1));

			// (2*sigma12 + C2)
			Image<Bgr, Single> temp2 = sigma12.ConvertScale<Single>(2, 0);
			temp2 = temp2.Add(new Bgr(C2, C2, C2));

			// ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
			Image<Bgr, Single> temp3 = temp1.Mul(temp2);

			// (mu1_sq + mu2_sq + C1)
			temp1 = mu1_sq.Add(mu2_sq);
			temp1 = temp1.Add(new Bgr(C1, C1, C1));

			// (sigma1_sq + sigma2_sq + C2)
			temp2 = sigma1_sq.Add(sigma12);
			temp2 = temp2.Add(new Bgr(C2, C2, C2));

			// ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
			temp1 = temp1.Mul(temp2, 1);

			// ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
			Image<Bgr, Single> ssim_map = new Image<Bgr, float>(imageSize);
			CvInvoke.Divide(temp3, temp1, ssim_map);

			Bgr avg = new Bgr();
			MCvScalar sdv = new MCvScalar();
			ssim_map.AvgSdv(out avg, out sdv);

			SSIM[(int)RGBIndex.Red] = avg.Red;
			SSIM[(int)RGBIndex.Green] = avg.Green;
			SSIM[(int)RGBIndex.Blue] = avg.Blue;
			SSIM[(int)RGBIndex.All] = avg.Red * avg.Green * avg.Blue;

			if (SSIM[(int)RGBIndex.All] == 1)//Same Image
			{
				NumDifferences = 0;
				return SSIM[(int)RGBIndex.All];
			}
			Image<Gray, Single> gray32 = new Image<Gray, float>(imageSize);
			CvInvoke.CvtColor(ssim_map, gray32, ColorConversion.Bgr2Gray);

			Image<Gray, Byte> gray8 = gray32.ConvertScale<Byte>(255, 0);
			Image<Gray, Byte> gray1 = gray8.ThresholdBinaryInv(new Gray(254), new Gray(255));

			VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
			CvInvoke.FindContours(gray1, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

			NumDifferences = contours.Size;
			for (int i = 0; i < NumDifferences; i++)
			{
				using (VectorOfPoint contour = contours[i])
				{
					Rectangle rect = CvInvoke.BoundingRectangle(contour);
					diff.Draw(rect, new Bgr(RectColor), 2);
				}
			}

			if (ImageDifferent == null)
				return SSIM[(int)RGBIndex.All];

			try
			{
				diff.Save(ImageDifferent);
			}
			catch (Exception ex)
			{
				throw new ImageDiffException(ex.Message);
			}
			return SSIM[(int)RGBIndex.All];
		}
    }
}
